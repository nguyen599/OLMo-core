import pytest
import torch
from torch.distributed.tensor import DTensor, Shard, init_device_mesh

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import (
    CuTeRMSNorm,
    FusedRMSNorm,
    L2Norm,
    LayerNormConfig,
    LayerNormType,
    QwenRMSNorm,
    RMSNorm,
)
from olmo_core.nn import layer_norm as layer_norm_mod
from olmo_core.testing import BACKENDS, requires_flash_attn_2, requires_gpu, requires_quack, run_distributed_test
from olmo_core.utils import get_default_device, seed_all


class FakeTENormKernels:
    @staticmethod
    def rmsnorm_fwd(x, weight, eps, quantizer, output_quantizer, te_dtype, sm_margin, zero_centered_gamma):
        del quantizer, output_quantizer, te_dtype, sm_margin
        effective_weight = weight + 1 if zero_centered_gamma else weight
        rstdev = torch.rsqrt(x.pow(2).mean(dim=-1) + eps)
        y = x * rstdev.unsqueeze(-1) * effective_weight
        return y, None, rstdev

    @staticmethod
    def rmsnorm_bwd(dy, x, rstdev, weight, sm_margin, zero_centered_gamma):
        del sm_margin
        effective_weight = weight + 1 if zero_centered_gamma else weight
        grad_normed = dy * effective_weight
        inner_dim = x.shape[-1]
        correction = (grad_normed * x).sum(dim=-1, keepdim=True) / inner_dim
        dx = grad_normed * rstdev.unsqueeze(-1) - x * rstdev.pow(3).unsqueeze(-1) * correction
        dw = (dy * x * rstdev.unsqueeze(-1)).sum(dim=0)
        return dx, dw


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("bias", [pytest.param(True, id="bias"), pytest.param(False, id="no-bias")])
@pytest.mark.parametrize(
    "dtype", [pytest.param(torch.float32, id="fp32"), pytest.param(torch.bfloat16, id="bf16")]
)
def test_fused_rms_norm(bias, dtype):
    dim = 64
    norm = RMSNorm(size=dim, bias=bias, init_device="cuda")
    norm_fused = FusedRMSNorm(size=dim, bias=bias, init_device="cuda")

    x = torch.randn(4, dim, device="cuda", dtype=dtype)
    y1 = norm(x)
    y2 = norm_fused(x)
    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_quack
def test_cute_rms_norm():
    dim = 64
    norm = CuTeRMSNorm(size=dim, init_device="cuda")
    norm.compile()
    ref_norm = RMSNorm(size=dim, init_device="cuda")

    x = torch.randn(4, dim, requires_grad=True, device="cuda", dtype=torch.bfloat16)
    x_ref = x.detach().clone().requires_grad_(True)
    y = norm(x)
    y_ref = ref_norm(x_ref)
    torch.testing.assert_close(y, y_ref)

    y.sum().backward()
    y_ref.sum().backward()
    assert x.grad is not None
    assert x_ref.grad is not None
    torch.testing.assert_close(x.grad, x_ref.grad)


@pytest.mark.parametrize("norm_cls", [RMSNorm, QwenRMSNorm])
def test_te_rms_norm_matches_olmo_rms_norm(monkeypatch, norm_cls):
    monkeypatch.setattr(
        layer_norm_mod,
        "_load_transformer_engine_norm_kernels",
        lambda: (FakeTENormKernels, {torch.float32: object(), torch.bfloat16: object()}),
    )
    dim = 64
    norm = norm_cls(size=dim, bias=False, init_device="cpu")
    te_norm = norm_cls(size=dim, bias=False, init_device="cpu")
    te_norm.load_state_dict(norm.state_dict())
    te_norm.enable_te_rms_norm()

    x = torch.randn(4, 8, dim, requires_grad=True)
    te_x = x.detach().clone().requires_grad_(True)

    y = norm(x)
    te_y = te_norm(te_x)
    torch.testing.assert_close(te_y, y)

    y.square().sum().backward()
    te_y.square().sum().backward()
    torch.testing.assert_close(te_x.grad, x.grad)
    torch.testing.assert_close(te_norm.weight.grad, norm.weight.grad)


def test_te_rms_norm_rejects_bias(monkeypatch):
    monkeypatch.setattr(
        layer_norm_mod,
        "_load_transformer_engine_norm_kernels",
        lambda: (FakeTENormKernels, {torch.float32: object(), torch.bfloat16: object()}),
    )
    norm = RMSNorm(size=64, bias=True, init_device="cpu")
    with pytest.raises(OLMoConfigurationError):
        norm.enable_te_rms_norm()


def _run_dtensor_te_rms_norm(state_path, inputs_path, outputs_path):
    layer_norm_mod._load_transformer_engine_norm_kernels = lambda: (
        FakeTENormKernels,
        {torch.float32: object(), torch.bfloat16: object()},
    )
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))
    norm = QwenRMSNorm(size=16, bias=False, init_device=device.type)
    norm.load_state_dict(torch.load(state_path, map_location=device))
    norm.enable_te_rms_norm()

    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    local_x = x.chunk(world_size, dim=1)[rank].contiguous().requires_grad_(True)
    dt_x = DTensor.from_local(
        local_x,
        mesh["tp"],
        (Shard(1),),
        run_check=False,
        shape=x.shape,
        stride=x.stride(),
    )
    y = norm(dt_x).to_local()
    y.sum().backward()

    y_ref = torch.load(outputs_path, map_location=device).chunk(world_size, dim=1)[rank]
    torch.testing.assert_close(y, y_ref)
    assert local_x.grad is not None


@pytest.mark.parametrize("backend", BACKENDS)
def test_dtensor_te_rms_norm_keeps_sequence_sharding(backend: str, tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")
    seed_all(0)
    norm = QwenRMSNorm(size=16, bias=False, init_device=device.type)
    x = torch.randn(2, 8, 16, device=device)
    y = norm(x)

    state_path = tmp_path / "norm.pt"
    inputs_path = tmp_path / "x.pt"
    outputs_path = tmp_path / "y.pt"
    torch.save(norm.state_dict(), state_path)
    torch.save(x, inputs_path)
    torch.save(y, outputs_path)

    run_distributed_test(
        _run_dtensor_te_rms_norm,
        backend=backend,
        start_method="spawn",
        func_args=(state_path, inputs_path, outputs_path),
    )


def test_layer_norm_builder_config():
    norm = LayerNormConfig(name=LayerNormType.l2_norm).build(size=1024)
    assert isinstance(norm, L2Norm)

    with pytest.raises(OLMoConfigurationError):
        LayerNormConfig(name=LayerNormType.l2_norm, elementwise_affine=True).build(size=1024)
