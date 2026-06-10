import torch

from olmo_core.nn.attention import backend as backend_mod


class FakeTEDotProductAttention:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []

    def __call__(self, q, k, v, **kwargs):
        del k, v
        self.calls.append((q, kwargs))
        return q.clone()

    def set_context_parallel_group(self, **kwargs):
        self.cp_kwargs = kwargs


def _build_te_backend(monkeypatch):
    monkeypatch.setattr(backend_mod, "has_te_attn", lambda: True)
    monkeypatch.setattr(backend_mod, "TEDotProductAttention", FakeTEDotProductAttention)
    return backend_mod.TEAttentionBackend(head_dim=4, n_heads=2, n_kv_heads=2)


def test_te_backend_forwards_bshd_when_no_document_lengths(monkeypatch):
    attn = _build_te_backend(monkeypatch)
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn(2, 3, 2, 4)
    v = torch.randn(2, 3, 2, 4)

    out = attn((q, k, v))

    assert out.shape == q.shape
    call_q, kwargs = attn.te_attn.calls[-1]
    assert call_q.shape == q.shape
    assert "qkv_format" not in kwargs
    assert kwargs["cu_seqlens_q"] is None
    assert kwargs["cu_seqlens_kv"] is None


def test_te_backend_uses_thd_for_packed_document_lengths(monkeypatch):
    attn = _build_te_backend(monkeypatch)
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn(2, 3, 2, 4)
    v = torch.randn(2, 3, 2, 4)
    cu_doc_lens = torch.tensor([0, 3, 6], dtype=torch.int32)

    out = attn((q, k, v), cu_doc_lens=cu_doc_lens, max_doc_len=3)

    assert out.shape == q.shape
    call_q, kwargs = attn.te_attn.calls[-1]
    assert call_q.shape == (6, 2, 4)
    assert kwargs["qkv_format"] == "thd"
    assert kwargs["attn_mask_type"] == "padding_causal"
    assert kwargs["cu_seqlens_q"] is cu_doc_lens
    assert kwargs["cu_seqlens_kv"] is cu_doc_lens
    assert kwargs["cu_seqlens_q_padded"] is cu_doc_lens
    assert kwargs["cu_seqlens_kv_padded"] is cu_doc_lens
    assert kwargs["max_seqlen_q"] == 3
    assert kwargs["max_seqlen_kv"] == 3
