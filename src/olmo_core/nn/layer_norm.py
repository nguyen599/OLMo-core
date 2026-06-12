from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate

from ..config import DType, StrEnum
from ..exceptions import OLMoConfigurationError
from .config import ModuleConfig
from .functional import l2_normalize

__all__ = [
    "LayerNormType",
    "LayerNormConfig",
    "LayerNorm",
    "RMSNorm",
    "QwenRMSNorm",
    "CuTeRMSNorm",
    "FusedRMSNorm",
    "LigerRMSNorm",
    "LigerMegatronRMSNorm",
    "L2Norm",
]


def _load_transformer_engine_norm_kernels() -> tuple[object, dict[torch.dtype, object]] | None:
    try:
        import transformer_engine_torch as tex  # type: ignore
        from transformer_engine.pytorch.constants import TE_DType  # type: ignore
    except Exception:
        return None
    return tex, TE_DType


class _TERMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float, full_precision: bool):
        loaded = _load_transformer_engine_norm_kernels()
        if loaded is None:
            raise OLMoConfigurationError(
                "Transformer Engine RMSNorm kernels are not available. "
                "Install Transformer Engine or disable TE layer norm mode."
            )
        tex, te_dtype = loaded
        input_shape = x.shape
        weight_shape = weight.shape
        inner_dim = math.prod(weight_shape)
        og_dtype = x.dtype
        compute_dtype = torch.float32 if full_precision else x.dtype
        x_compute = x.contiguous().to(compute_dtype).view((-1, inner_dim))
        weight_compute = weight.to(compute_dtype).view((inner_dim,))
        y, _, rstdevs = tex.rmsnorm_fwd(
            x_compute,
            weight_compute,
            eps,
            None,
            None,
            te_dtype[compute_dtype],
            0,
            False,
        )
        ctx.save_for_backward(x_compute, weight_compute, rstdevs)
        ctx.input_shape = input_shape
        ctx.weight_shape = weight_shape
        ctx.input_dtype = og_dtype
        ctx.weight_dtype = weight.dtype
        ctx.compute_dtype = compute_dtype
        return y.view(input_shape).to(og_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        loaded = _load_transformer_engine_norm_kernels()
        if loaded is None:
            raise OLMoConfigurationError(
                "Transformer Engine RMSNorm kernels are not available during backward."
            )
        tex, _ = loaded
        x, weight, rstdevs = ctx.saved_tensors
        dy = grad_output.contiguous().to(ctx.compute_dtype).view(x.size())
        dx, dw = tex.rmsnorm_bwd(dy, x, rstdevs, weight, 0, False)
        return (
            dx.view(ctx.input_shape).to(ctx.input_dtype),
            dw.view(ctx.weight_shape).to(ctx.weight_dtype),
            None,
            None,
        )


def _te_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    full_precision: bool,
) -> torch.Tensor:
    if isinstance(x, DTensor):
        local_weight = weight.to_local() if isinstance(weight, DTensor) else weight
        local_out = _TERMSNormFunction.apply(x.to_local(), local_weight, eps, full_precision)
        return DTensor.from_local(
            local_out,
            x.device_mesh,
            x.placements,
            run_check=False,
            shape=x.shape,
            stride=x.stride(),
        )
    return _TERMSNormFunction.apply(x, weight, eps, full_precision)


def _load_liger_rms_norm_function() -> object | None:
    try:
        import torch.distributed.tensor  # noqa: F401
        from liger_kernel.ops import LigerRMSNormFunction  # type: ignore
    except Exception:
        return None
    return LigerRMSNormFunction


def _load_liger_megatron_rms_norm_class() -> object | None:
    try:
        from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm  # type: ignore
    except Exception:
        return None
    return LigerMegatronRMSNorm


def _dtensor_has_partial_placement(x: DTensor) -> bool:
    return any("Partial" in placement.__class__.__name__ for placement in x.placements)


def _dtensor_replicated_placements(x: DTensor) -> tuple[Replicate, ...]:
    return tuple(Replicate() for _ in x.placements)


def _wrap_liger_dtensor_output(local_out: torch.Tensor, x: DTensor) -> torch.Tensor:
    if isinstance(local_out, DTensor):
        return local_out

    out_shape = tuple(local_out.shape)
    x_shape = tuple(x.shape)
    x_local_shape = tuple(x.to_local().shape)
    replicated_placements = _dtensor_replicated_placements(x)

    if out_shape == x_shape:
        replicated = DTensor.from_local(
            local_out,
            x.device_mesh,
            replicated_placements,
            run_check=False,
            shape=x_shape,
            stride=local_out.stride(),
        )
        if _dtensor_has_partial_placement(x):
            return replicated
        return replicated.redistribute(device_mesh=x.device_mesh, placements=x.placements)

    if out_shape == x_local_shape:
        return DTensor.from_local(
            local_out,
            x.device_mesh,
            x.placements,
            run_check=False,
            shape=x.shape,
            stride=x.stride(),
        )

    raise OLMoConfigurationError(
        f"Liger RMSNorm returned shape {out_shape}, expected global DTensor shape {x_shape} "
        f"or local DTensor shape {x_local_shape}."
    )


@torch.compiler.disable(recursive=False)
def _liger_ops_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    casting_mode: str,
) -> torch.Tensor:
    liger_fn = _load_liger_rms_norm_function()
    if liger_fn is None:
        raise OLMoConfigurationError(
            "Liger RMSNorm kernels are not available. Install liger-kernel or disable "
            "Liger layer norm mode."
        )

    if isinstance(x, DTensor):
        local_weight = weight.full_tensor() if isinstance(weight, DTensor) else weight
        local_out = liger_fn.apply(x, local_weight, eps, 0.0, casting_mode, False, None)
        return _wrap_liger_dtensor_output(local_out, x)

    return liger_fn.apply(x, weight, eps, 0.0, casting_mode, False, None)


@torch.compiler.disable(recursive=False)
def _liger_megatron_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    liger_cls = _load_liger_megatron_rms_norm_class()
    if liger_cls is None:
        raise OLMoConfigurationError(
            "Liger RMSNorm kernels are not available. Install liger-kernel or disable "
            "Liger layer norm mode."
        )
    # Call through Liger's Megatron-Core integration path. A light proxy lets
    # OLMo keep its own checkpoint parameter names while using the upstream
    # Megatron wrapper's forward implementation.
    proxy = SimpleNamespace(weight=weight, eps=eps, _offset=0.0)

    if isinstance(x, DTensor):
        local_x = x.full_tensor()
        local_weight = weight.full_tensor() if isinstance(weight, DTensor) else weight
        proxy.weight = local_weight
        local_out = liger_cls.forward(proxy, local_x)
        return _wrap_liger_dtensor_output(local_out, x)

    return liger_cls.forward(proxy, x)


class LayerNormType(StrEnum):
    """
    An enumeration of the different layer norm implementations.
    """

    default = "default"
    """
    ➡️ :class:`LayerNorm`
    """
    rms = "rms"
    """
    ➡️ :class:`RMSNorm`
    """
    qwen_rms = "qwen_rms"
    """
    ➡️ :class:`QwenRMSNorm`
    """
    cute_rms = "cute_rms"
    """
    ➡️ :class:`CuTeRMSNorm`
    """
    fused_rms = "fused_rms"
    """
    ➡️ :class:`FusedRMSNorm`
    """
    liger_rms = "liger_rms"
    """
    ➡️ :class:`LigerRMSNorm`
    """
    liger_megatron_rms = "liger_megatron_rms"
    """
    ➡️ :class:`LigerMegatronRMSNorm`
    """
    l2_norm = "l2_norm"
    """
    ➡️ :class:`L2Norm`
    """


@dataclass
class LayerNormConfig(ModuleConfig):
    """
    A config for conveniently building any one of the different layer norm classes.

    See the :class:`LayerNorm` subclasses to learn which fields are valid for each implementation.
    """

    name: LayerNormType = LayerNormType.default
    """
    The name of the implementation.
    """
    eps: Optional[float] = None
    elementwise_affine: Optional[bool] = None
    bias: Optional[bool] = None
    full_precision: Optional[bool] = None
    dtype: Optional[DType] = None

    def num_params(self, size: int) -> int:
        """
        The number of parameters in the module once built.

        :param size: The size of the input along the dimension to be normalized.
        """
        elementwise_affine = (
            self.elementwise_affine
            if self.elementwise_affine is not None
            else self.name != LayerNormType.l2_norm
        )
        bias = self.bias if self.bias is not None else self.name != LayerNormType.l2_norm
        ln_params = 0
        if elementwise_affine:
            ln_params += size
            if bias:
                ln_params += size
        return ln_params

    def build(self, size: int, init_device: str = "cpu") -> "LayerNorm":
        """
        Construct the corresponding LayerNorm class.

        :param size: The size of the input along the dimension to be normalized.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        if (dtype := kwargs.pop("dtype", None)) is not None:
            kwargs.update(dtype=dtype.as_pt())

        try:
            if self.name == LayerNormType.default:
                return LayerNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.rms:
                return RMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.qwen_rms:
                return QwenRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.cute_rms:
                return CuTeRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.fused_rms:
                return FusedRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.liger_rms:
                return LigerRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.liger_megatron_rms:
                return LigerMegatronRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.l2_norm:
                return L2Norm(size=size, **kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class LayerNorm(nn.Module):
    """
    Layer normalization.

    :param size: The size of the input along the dimension to be normalized.
    :param eps: The epsilon used for numerical stability.
    :param elementwise_affine: Whether to include an element-wise affine transform.
    :param bias: Whether the element-wise affine should include an element-wise bias.
        Ignored if ``elementwise_affine=False``.
    :param full_precision: Force the operation to run in full precision regardless of the input
        data type.
    :param dtype: The default data type to use for the weight and bias in the element-wise affine.
        If ``full_precision=False`` it can be useful to set this to the expected input data type.
        Ignored if ``elementwise_affine=False``.
    :param init_device: The device used when initializing the element-wise weight/bias.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.normalized_shape = (size,)
        self.eps = eps
        self.full_precision = full_precision
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, dtype=dtype, device=init_device)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, dtype=dtype, device=init_device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)
        self._te_rms_norm_enabled = False
        self._liger_rms_norm_enabled = False
        self._liger_rms_norm_casting_mode = "gemma" if full_precision else "none"
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def extra_repr(self):
        if self.weight is not None and self.bias is not None:
            return f"{tuple(self.weight.shape)}, bias=True, eps={self.eps}"
        elif self.weight is not None:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        else:
            return f"eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm.

        :param x: The input.
        """
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            x = F.layer_norm(
                x,
                self.normalized_shape,
                weight=None if self.weight is None else self.weight.type_as(x),
                bias=None if self.bias is None else self.bias.type_as(x),
                eps=self.eps,
            )

            return x.to(og_dtype)


class RMSNorm(LayerNorm):
    """
    RMSNorm, a simplified layer norm implementation.
    """

    def enable_te_rms_norm(self) -> None:
        if self.weight is None:
            raise OLMoConfigurationError(
                "Transformer Engine RMSNorm requires elementwise_affine=True."
            )
        if self.bias is not None:
            raise OLMoConfigurationError("Transformer Engine RMSNorm does not support bias.")
        if _load_transformer_engine_norm_kernels() is None:
            raise OLMoConfigurationError(
                "Transformer Engine RMSNorm kernels are not available. "
                "Install Transformer Engine or disable TE layer norm mode."
            )
        self._te_rms_norm_enabled = True

    def enable_liger_rms_norm(self, casting_mode: Optional[str] = None) -> None:
        if self.weight is None:
            raise OLMoConfigurationError("Liger RMSNorm requires elementwise_affine=True.")
        if self.bias is not None:
            raise OLMoConfigurationError("Liger RMSNorm does not support bias.")
        if _load_liger_rms_norm_function() is None:
            raise OLMoConfigurationError(
                "Liger RMSNorm kernels are not available. Install liger-kernel or disable "
                "Liger layer norm mode."
            )
        self._liger_rms_norm_casting_mode = casting_mode or ("gemma" if self.full_precision else "none")
        self._liger_rms_norm_enabled = True

    def enable_liger_megatron_rms_norm(self, casting_mode: Optional[str] = None) -> None:
        if self.weight is None:
            raise OLMoConfigurationError("Liger Megatron RMSNorm requires elementwise_affine=True.")
        if self.bias is not None:
            raise OLMoConfigurationError("Liger Megatron RMSNorm does not support bias.")
        if casting_mode not in (None, "llama"):
            raise OLMoConfigurationError(
                "The Liger Megatron RMSNorm path only exposes Llama-style casting."
            )
        if _load_liger_megatron_rms_norm_class() is None:
            raise OLMoConfigurationError(
                "Liger RMSNorm kernels are not available. Install liger-kernel or disable "
                "Liger layer norm mode."
            )
        self._liger_rms_norm_casting_mode = "megatron"
        self._liger_rms_norm_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        :param x: The input.
        """
        if self._te_rms_norm_enabled:
            assert self.weight is not None
            return _te_rms_norm(x, self.weight, self.eps, self.full_precision)
        if self._liger_rms_norm_enabled:
            assert self.weight is not None
            if self._liger_rms_norm_casting_mode == "megatron":
                return _liger_megatron_rms_norm(x, self.weight, self.eps)
            return _liger_ops_rms_norm(x, self.weight, self.eps, self._liger_rms_norm_casting_mode)

        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)

            if self.weight is not None:
                if self.bias is not None:
                    x = self.weight.type_as(x) * x + self.bias.type_as(x)
                else:
                    x = self.weight.type_as(x) * x

            return x.to(og_dtype)


class QwenRMSNorm(RMSNorm):
    """
    RMSNorm variant matching HuggingFace's ``Qwen3RMSNorm`` rounding order: the input is
    cast back to its original dtype before being multiplied by the affine weight, so the
    weight multiply happens in the input dtype rather than fp32.
    """

    def enable_liger_rms_norm(self, casting_mode: Optional[str] = None) -> None:
        super().enable_liger_rms_norm(casting_mode or "llama")

    def enable_liger_megatron_rms_norm(self, casting_mode: Optional[str] = None) -> None:
        super().enable_liger_megatron_rms_norm(casting_mode or "llama")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._te_rms_norm_enabled:
            assert self.weight is not None
            return _te_rms_norm(x, self.weight, self.eps, self.full_precision)
        if self._liger_rms_norm_enabled:
            assert self.weight is not None
            if self._liger_rms_norm_casting_mode == "megatron":
                return _liger_megatron_rms_norm(x, self.weight, self.eps)
            return _liger_ops_rms_norm(x, self.weight, self.eps, self._liger_rms_norm_casting_mode)

        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

            if self.weight is not None:
                x = x * self.weight.type_as(x)
            if self.bias is not None:
                x = x + self.bias.type_as(x)

            return x


class CuTeRMSNorm(RMSNorm):
    """
    A CuTe-based implementation from the QuACK library.

    .. warning::
        This requires `quack <https://github.com/Dao-AILab/quack>`_ to be installed.

    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from quack import rmsnorm as rms_norm_fn  # type: ignore

        if not full_precision:
            # the CUTE kernel always casts to full precision internally
            raise NotImplementedError(
                f"Currently only 'full_precision=True' is supported with '{self.__class__.__name__}'"
            )

        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self._rms_norm_fn = rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms_norm_fn(
            x,
            weight=None if self.weight is None else self.weight.type_as(x),
            bias=None if self.bias is None else self.bias.type_as(x),
            eps=self.eps,
        ).to(x.dtype)


class FusedRMSNorm(RMSNorm):
    """
    A "fused" triton-based implementation of :class:`RMSNorm`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.

    .. warning::
        Currently only ``elementwise_affine=True`` is supported.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from flash_attn.ops.triton.layer_norm import rms_norm_fn  # type: ignore

        if not elementwise_affine:
            raise NotImplementedError(
                f"Currently only 'elementwise_affine=True' is supported with '{self.__class__.__name__}'"
            )
        if not full_precision:
            # the triton kernel always casts to full precision internally
            raise NotImplementedError(
                f"Currently only 'full_precision=True' is supported with '{self.__class__.__name__}'"
            )

        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self._rms_norm_fn = rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        og_dtype = x.dtype
        if self.full_precision:
            x = x.float()
        return self._rms_norm_fn(
            x,
            self.weight.type_as(x),
            None if self.bias is None else self.bias.type_as(x),
            eps=self.eps,
        ).to(og_dtype)


class LigerRMSNorm(RMSNorm):
    """
    A Liger Triton-kernel implementation of :class:`RMSNorm`.

    This calls through the normal ``liger_kernel.ops.LigerRMSNormFunction``
    path while preserving OLMo checkpoint parameter names.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self.enable_liger_rms_norm("gemma" if full_precision else "none")


class LigerMegatronRMSNorm(RMSNorm):
    """
    A Liger Megatron-Core RMSNorm implementation that calls through
    ``liger_kernel.megatron.rms_norm.LigerMegatronRMSNorm`` while preserving
    OLMo checkpoint parameter names.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self.enable_liger_megatron_rms_norm("llama")


class L2Norm(LayerNorm):
    """
    A variant of layer norm that just normalizes the last dimension of the input by its L2 norm,
    as done in nGPT.

    :param size: The size of the input along the dimension to be normalized.
    """

    def __init__(
        self,
        *,
        size: int,
    ):
        super().__init__(size=size, elementwise_affine=False, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(x)
