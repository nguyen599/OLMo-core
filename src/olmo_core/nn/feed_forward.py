import functools
import math
import os
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from ..config import DType, StrEnum
from ..doc_utils import beta_feature
from ..exceptions import OLMoConfigurationError
from .config import ModuleConfig
from .functional import l2_normalize
from .utils import get_tp_wrappers

__all__ = [
    "ActivationFunction",
    "FeedForwardType",
    "FeedForwardConfig",
    "FeedForward",
    "NormalizedFeedForward",
]


_TE_LINEAR_NAMES = ("w1", "w2", "w3")
log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rank_allowed(spec: str, rank: str, local_rank: str) -> bool:
    if spec.strip().lower() in {"", "all", "*"}:
        return True
    wanted = {part.strip() for part in spec.split(",") if part.strip()}
    return rank in wanted or f"rank:{rank}" in wanted or f"local:{local_rank}" in wanted


def _cuda_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    active = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_active = torch.cuda.max_memory_allocated(device)
    gib = 1024**3
    return (
        f"device={device} active={active / gib:.2f}GiB reserved={reserved / gib:.2f}GiB "
        f"peak_active={peak_active / gib:.2f}GiB free={free / gib:.2f}GiB total={total / gib:.2f}GiB"
    )


def _restore_chunked_dtensor_layout(
    x: DTensor,
    out: torch.Tensor | DTensor,
    token_dim: int,
) -> DTensor:
    placements = tuple(x.placements)
    shard_placement = next((placement for placement in placements if isinstance(placement, Shard)), None)
    if shard_placement is None:
        if isinstance(out, DTensor):
            return out.redistribute(device_mesh=x.device_mesh, placements=placements)
        return DTensor.from_local(out, x.device_mesh, placements, run_check=False)

    local_out = out.to_local() if isinstance(out, DTensor) else out
    local_x = x.to_local()
    if local_out.shape[token_dim] != local_x.shape[token_dim]:
        shard_dim = shard_placement.dim
        shard_dim = shard_dim if shard_dim >= 0 else local_out.dim() + shard_dim
        local_rank = x.device_mesh.get_local_rank()
        local_size = local_x.shape[shard_dim]
        start = local_rank * local_size
        local_out = local_out.narrow(shard_dim, start, local_size).contiguous()
    return DTensor.from_local(local_out, x.device_mesh, placements, run_check=False)


def _load_te_linear_cls() -> Optional[Type[nn.Module]]:
    try:
        from transformer_engine.pytorch import Linear as TELinear  # type: ignore
    except Exception:
        return None
    return TELinear


def _load_transformer_engine_torch() -> object | None:
    try:
        import transformer_engine_torch as tex  # type: ignore
    except Exception:
        return None
    return tex


class _TEGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, linear: torch.Tensor, activation):
        tex = _load_transformer_engine_torch()
        if tex is None:
            raise OLMoConfigurationError(
                "transformer_engine_torch is not available. Install Transformer Engine or disable "
                "TE fused feed-forward activation mode."
            )
        glu_input = torch.cat((gate, linear), dim=-1).contiguous()
        if activation == ActivationFunction.silu:
            output = tex.swiglu(glu_input, None)
        else:
            raise OLMoConfigurationError(f"Unsupported TE fused feed-forward activation: {activation}")
        ctx.save_for_backward(glu_input)
        ctx.activation = activation
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        tex = _load_transformer_engine_torch()
        if tex is None:
            raise OLMoConfigurationError(
                "transformer_engine_torch is not available during TE fused feed-forward backward."
            )
        (glu_input,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if ctx.activation == ActivationFunction.silu:
            grad_glu_input = tex.dswiglu(grad_output, glu_input, None)
        else:
            raise OLMoConfigurationError(
                f"Unsupported TE fused feed-forward activation: {ctx.activation}"
            )
        gate_grad, linear_grad = grad_glu_input.chunk(2, dim=-1)
        return gate_grad, linear_grad, None


def _strip_te_linear_extra_state(
    module: nn.Module,
    state_dict: dict[str, object],
    prefix: str,
    local_metadata: dict[str, object],
) -> None:
    del module, local_metadata
    for name in _TE_LINEAR_NAMES:
        state_dict.pop(f"{prefix}{name}._extra_state", None)


def _restore_te_linear_extra_state_for_load(
    module: nn.Module,
    state_dict: dict[str, object],
    prefix: str,
    local_metadata: dict[str, object],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    for name in _TE_LINEAR_NAMES:
        linear = getattr(module, name, None)
        if linear is None:
            continue
        extra_state_key = f"{prefix}{name}._extra_state"
        if extra_state_key in state_dict:
            continue
        if type(linear).get_extra_state is not nn.Module.get_extra_state:
            state_dict[extra_state_key] = linear.get_extra_state()


def _copy_linear_weights(src: nn.Module, dst: nn.Module) -> None:
    src_weight = getattr(src, "weight")
    dst_weight = getattr(dst, "weight")
    if src_weight.device.type == "meta" or dst_weight.device.type == "meta":
        return
    with torch.no_grad():
        dst_weight.copy_(src_weight)
        src_bias = getattr(src, "bias", None)
        dst_bias = getattr(dst, "bias", None)
        if src_bias is not None and dst_bias is not None:
            dst_bias.copy_(src_bias)


def _build_te_linear_from_linear(linear: nn.Linear) -> nn.Module:
    te_linear_cls = _load_te_linear_cls()
    if te_linear_cls is None:
        raise OLMoConfigurationError(
            "Transformer Engine Linear is not available. Install transformer_engine or disable "
            "TE feed-forward mode."
        )
    te_linear = te_linear_cls(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        params_dtype=linear.weight.dtype,
        device=linear.weight.device,
        save_original_input=True,
    )
    _copy_linear_weights(linear, te_linear)
    return te_linear


class ActivationFunction(StrEnum):
    """
    An enumeration of the supported activation functions for feed-forward modules.
    """

    silu = "silu"
    """
    SiLU/Swish activation function, used for SwiGLU.
    """

    gelu_tanh = "gelu_tanh"
    """
    GELU with tanh approximation, used for GeGLU.
    """

    def build(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self == ActivationFunction.silu:
            return F.silu
        elif self == ActivationFunction.gelu_tanh:
            return functools.partial(F.gelu, approximate="tanh")
        else:
            raise NotImplementedError(self)


class FeedForwardType(StrEnum):
    """
    An enumeration of the different feed-forward / MLP implementations.
    """

    default = "default"
    """
    ➡️ :class:`FeedForward`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedFeedForward`
    """


@dataclass
class FeedForwardConfig(ModuleConfig):
    """
    A config for building :class:`FeedForward` modules.
    """

    hidden_size: int
    name: FeedForwardType = FeedForwardType.default
    """
    The name of the implementation.
    """
    bias: Optional[bool] = None
    dtype: Optional[DType] = None
    activation: ActivationFunction = ActivationFunction.silu
    """
    The activation function to use. See :class:`ActivationFunction` for options.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        bias = self.bias if self.bias is not None else self.name != FeedForwardType.normalized

        params = 0

        params += 3 * d_model * self.hidden_size
        if bias:
            params += 2 * self.hidden_size + d_model

        # w1 + w3 scaling factors
        if self.name == FeedForwardType.normalized:
            params += 2 * self.hidden_size

        return params

    def build(
        self, d_model: int, *, dtype: Optional[torch.dtype] = None, init_device: str = "cpu"
    ) -> "FeedForward":
        """
        Build the corresponding feed-forward module.

        :param d_model: The model dimensionality.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            if self.name == FeedForwardType.default:
                return FeedForward(**kwargs)
            elif self.name == FeedForwardType.normalized:
                activation = kwargs.get("activation", ActivationFunction.silu)
                if activation != ActivationFunction.silu:
                    raise OLMoConfigurationError(
                        f"NormalizedFeedForward only supports 'silu' activation, got '{activation}'"
                    )
                return NormalizedFeedForward(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class FeedForward(nn.Module):
    """
    Basic feed-forward module with gated activation (SwiGLU or GeGLU).
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        activation: ActivationFunction = ActivationFunction.silu,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.activation = activation
        self.activation_fn = activation.build()
        self.w1 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.w2 = nn.Linear(hidden_size, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w3 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self._te_linear_enabled = False
        self._te_glu_enabled = False
        self._chunk_size_tokens = 0
        self._memory_profile_name: str | None = None
        self._memory_profile_calls = 0
        self._tp_mesh: DeviceMesh | None = None
        self._tp_output_layout: Placement | None = None

    def enable_te_linear(self) -> None:
        """
        Replace the dense MLP projections with Transformer Engine Linear modules.

        The parameter names remain ``w1.weight``, ``w2.weight``, and ``w3.weight``, so
        existing non-TE checkpoints can still load. TE runtime extra state is omitted
        from checkpoints to keep them compatible with the standard OLMo-core layout.
        """
        if self._te_linear_enabled:
            return
        self.w1 = _build_te_linear_from_linear(self.w1)  # type: ignore[assignment]
        self.w2 = _build_te_linear_from_linear(self.w2)  # type: ignore[assignment]
        self.w3 = _build_te_linear_from_linear(self.w3)  # type: ignore[assignment]
        self.register_state_dict_post_hook(_strip_te_linear_extra_state)
        self.register_load_state_dict_pre_hook(_restore_te_linear_extra_state_for_load)
        self._te_linear_enabled = True

    def enable_te_glu(self) -> None:
        """
        Use Transformer Engine's fused GLU activation kernel for the MLP middle op.

        This keeps the existing OLMo-core linear modules and therefore stays compatible
        with DTensor TP sharding and checkpoint loading. It only replaces the local
        ``activation(w1(x)) * w3(x)`` operation.
        """
        if self.__class__ is not FeedForward:
            raise OLMoConfigurationError(
                "TE fused feed-forward activation is only implemented for the default FeedForward."
            )
        if self._te_linear_enabled:
            raise OLMoConfigurationError(
                "TE fused feed-forward activation cannot be combined with TE Linear feed-forward mode."
            )
        if self.activation != ActivationFunction.silu:
            raise OLMoConfigurationError(
                f"TE fused feed-forward activation currently supports only {ActivationFunction.silu}; "
                f"got {self.activation}."
            )
        if _load_transformer_engine_torch() is None:
            raise OLMoConfigurationError(
                "transformer_engine_torch is not available. Install Transformer Engine or disable "
                "TE fused feed-forward activation mode."
            )
        self._te_glu_enabled = True

    def enable_chunked_forward(self, chunk_size_tokens: int) -> None:
        """
        Split the local token dimension during the feed-forward projection.

        This reduces the peak memory from the two hidden-size intermediate tensors
        in ``w1`` and ``w3`` at the cost of launching more GEMMs.
        """
        if chunk_size_tokens < 0:
            raise OLMoConfigurationError("Feed-forward chunk size must be >= 0.")
        if chunk_size_tokens > 0 and self._tp_mesh is not None and self._tp_mesh.size() > 1:
            raise OLMoConfigurationError(
                "Feed-forward token chunking is not currently safe with TP>1."
            )
        self._chunk_size_tokens = chunk_size_tokens

    def set_memory_profile_name(self, name: str) -> None:
        self._memory_profile_name = name

    def _should_log_memory_profile(self) -> bool:
        if not _env_flag("OLMO_FF_MEMORY_PROFILE"):
            return False
        rank = os.environ.get("RANK", os.environ.get("GLOBAL_RANK", "0"))
        local_rank = os.environ.get("LOCAL_RANK", "0")
        ranks = os.environ.get("OLMO_FF_MEMORY_PROFILE_RANKS", "all")
        if not _rank_allowed(ranks, rank, local_rank):
            return False
        max_calls = int(os.environ.get("OLMO_FF_MEMORY_PROFILE_MAX_CALLS", "1"))
        return max_calls <= 0 or self._memory_profile_calls < max_calls

    def _log_memory_profile(self, point: str, x: torch.Tensor) -> None:
        if not self._should_log_memory_profile():
            return
        if _env_flag("OLMO_FF_MEMORY_PROFILE_SYNC") and torch.cuda.is_available():
            torch.cuda.synchronize()
        rank = os.environ.get("RANK", os.environ.get("GLOBAL_RANK", "0"))
        local_rank = os.environ.get("LOCAL_RANK", "0")
        name = self._memory_profile_name or self.__class__.__name__
        log.warning(
            "FFN memory profile rank=%s local_rank=%s module=%s call=%d point=%s "
            "shape=%s chunk_size_tokens=%d %s",
            rank,
            local_rank,
            name,
            self._memory_profile_calls,
            point,
            tuple(x.shape),
            self._chunk_size_tokens,
            _cuda_memory_summary(),
        )

    def _forward_unchecked(self, x: torch.Tensor) -> torch.Tensor:
        self._log_memory_profile("before_w1", x)
        gate = self.w1(x)
        self._log_memory_profile("after_w1_before_w3", gate)
        linear = self.w3(x)
        self._log_memory_profile("after_w3_before_activation", linear)
        if self._te_glu_enabled:
            hidden = _TEGLUFunction.apply(gate, linear, self.activation)
        else:
            hidden = self.activation_fn(gate) * linear
        self._log_memory_profile("after_activation_before_w2", hidden)
        out = self.w2(hidden)
        self._log_memory_profile("after_w2", out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        chunk_size_tokens = self._chunk_size_tokens
        if chunk_size_tokens <= 0 or x.numel() == 0:
            try:
                return self._forward_unchecked(x)
            finally:
                if self._should_log_memory_profile():
                    self._memory_profile_calls += 1

        token_dim = -2 if x.dim() > 2 else 0
        if x.shape[token_dim] <= chunk_size_tokens:
            try:
                return self._forward_unchecked(x)
            finally:
                if self._should_log_memory_profile():
                    self._memory_profile_calls += 1

        outputs = []
        try:
            for chunk_idx, chunk in enumerate(x.split(chunk_size_tokens, dim=token_dim)):
                self._log_memory_profile(f"chunk_{chunk_idx}_start", chunk)
                outputs.append(self._forward_unchecked(chunk))
                self._log_memory_profile(f"chunk_{chunk_idx}_end", outputs[-1])
            out = torch.cat(outputs, dim=token_dim)
            if isinstance(x, DTensor):
                return _restore_chunked_dtensor_layout(x, out, token_dim)
            if isinstance(out, DTensor) and self._tp_mesh is not None and self._tp_output_layout is not None:
                out = out.redistribute(
                    device_mesh=self._tp_mesh,
                    placements=(self._tp_output_layout,),
                )
            return out
        finally:
            if self._should_log_memory_profile():
                self._memory_profile_calls += 1

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        if self._te_linear_enabled and tp_mesh.size() > 1:
            raise OLMoConfigurationError(
                "Transformer Engine feed-forward mode currently supports only TP=1. "
                "TE Linear's internal TP uses local-shard parameters, while OLMo-core's "
                "current checkpoint load path expects DTensor-sharded MLP weights."
            )
        if self._chunk_size_tokens > 0 and tp_mesh.size() > 1:
            raise OLMoConfigurationError(
                "Feed-forward token chunking is not currently safe with TP>1."
            )
        self._tp_mesh = tp_mesh
        self._tp_output_layout = output_layout
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "w1": colwise_parallel(),
                "w2": rowwise_parallel(
                    output_layouts=output_layout, use_local_output=use_local_output
                ),
                "w3": colwise_parallel(),
            },
        )

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward)
        return 6 * sum(p.numel() for p in self.parameters())


@beta_feature
class NormalizedFeedForward(FeedForward):
    """
    An nGPT feed-forward implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        activation: ActivationFunction = ActivationFunction.silu,
    ):
        if activation != ActivationFunction.silu:
            raise OLMoConfigurationError(
                f"NormalizedFeedForward only supports 'silu' activation, got '{activation}'"
            )
        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            dtype=dtype,
            init_device=init_device,
            bias=False,
            activation=activation,
        )
        self.sw_init_value = 1.0
        self.sw_init_scaling = 1.0
        self.sw1 = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=init_device))
        self.sw3 = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=init_device))
        self.sqrt_d_model = math.sqrt(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sw1)
        nn.init.ones_(self.sw3)
        with torch.no_grad():
            self.sw1.mul_(self.sw_init_scaling)
            self.sw3.mul_(self.sw_init_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sw1 = self.sw1 * ((self.sw_init_value / self.sw_init_scaling) * self.sqrt_d_model)
        sw3 = self.sw3 * (self.sw_init_value / self.sw_init_scaling)
        return self.w2(F.silu(sw1 * self.w1(x)) * (sw3 * self.w3(x)))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized feed-forward variant"
        )

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)
        self._normalize_matrix(self.w3.weight)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))
