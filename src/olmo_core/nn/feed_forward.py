import functools
import math
from dataclasses import dataclass
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Placement, Replicate

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


def _load_te_linear_cls() -> Optional[Type[nn.Module]]:
    try:
        from transformer_engine.pytorch import Linear as TELinear  # type: ignore
    except Exception:
        return None
    return TELinear


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
        self.activation_fn = activation.build()
        self.w1 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.w2 = nn.Linear(hidden_size, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w3 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self._te_linear_enabled = False

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        return self.w2(self.activation_fn(self.w1(x)) * self.w3(x))

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
