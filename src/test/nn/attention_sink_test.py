import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.transformer.init import InitMethod


def _build_attention(*, attention_sink: bool, attention_sink_init_value: float = -10.0):
    config = AttentionConfig(
        n_heads=4,
        n_kv_heads=2,
        backend=AttentionBackendName.torch,
        dtype=DType.float32,
        attention_sink=attention_sink,
        attention_sink_init_value=attention_sink_init_value,
    )
    att = config.build(64, layer_idx=0, n_layers=1, init_device="cpu")
    att.init_weights(
        init_method=InitMethod.normal,
        d_model=64,
        block_idx=0,
        num_blocks=1,
        generator=torch.Generator().manual_seed(7),
    )
    return att


def test_attention_sink_near_noop_matches_plain_attention():
    plain = _build_attention(attention_sink=False)
    sink = _build_attention(attention_sink=True, attention_sink_init_value=-1e4)
    sink.load_state_dict(
        {key: value for key, value in plain.state_dict().items() if not key.startswith("sinks.")},
        strict=False,
    )
    with torch.no_grad():
        sink.sinks.weight.fill_(-1e4)

    x = torch.randn(2, 5, 64)
    torch.testing.assert_close(sink(x), plain(x), atol=1e-7, rtol=1e-7)


def test_attention_sink_receives_gradients():
    sink = _build_attention(attention_sink=True, attention_sink_init_value=-2.0)
    x = torch.randn(2, 5, 64, requires_grad=True)
    sink(x).float().pow(2).mean().backward()

    assert sink.sinks is not None
    assert sink.sinks.weight.grad is not None
    assert torch.isfinite(sink.sinks.weight.grad).all()
    assert sink.sinks.weight.grad.abs().sum() > 0
