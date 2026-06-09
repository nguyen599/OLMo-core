import pytest

import olmo_core.nn.attention.flash_attn_api as flash_attn_api
from olmo_core.nn.attention.flash_attn_api import (
    has_flash_attn_4,
    is_flash_attn_4_compute_capability_supported,
)


@pytest.mark.parametrize(
    ("compute_capability", "expected"),
    [
        ((8, 0), False),
        ((8, 9), False),
        ((9, 0), True),
        ((9, 10), True),
        ((10, 0), True),
        ((11, 0), True),
        ((12, 0), False),
    ],
)
def test_flash_attn_4_compute_capability_support(compute_capability, expected):
    assert is_flash_attn_4_compute_capability_supported(compute_capability) is expected


@pytest.mark.parametrize(
    ("compute_capability", "expected"),
    [
        ((9, 0), True),
        ((10, 0), True),
        ((11, 0), True),
        ((12, 0), False),
    ],
)
def test_has_flash_attn_4_allows_sm90_plus_when_module_is_available(monkeypatch, compute_capability, expected):
    monkeypatch.setattr(flash_attn_api, "flash_attn_4", object())
    monkeypatch.setattr(flash_attn_api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flash_attn_api.torch.cuda, "get_device_capability", lambda: compute_capability)

    assert has_flash_attn_4() is expected
