import logging
from abc import abstractmethod
from typing import Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh

from olmo_core.config import StrEnum
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_cp2hp_qkvpacked,
    all_to_all_single_hp2cp,
)
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.buffer_cache import BufferCache

from .flash_attn_api import (
    dispatch_flash_attn,
    dispatch_flash_attn_3,
    dispatch_flash_attn_3_qkvpacked,
    dispatch_flash_attn_3_with_kvcache,
    dispatch_flash_attn_3_with_sink,
    dispatch_flash_attn_4,
    dispatch_flash_attn_qkvpacked,
    dispatch_flash_attn_with_kvcache,
    dispatch_ring_flash_attn,
    dispatch_ring_flash_attn_qkvpacked,
    has_flash_attn_2,
    has_flash_attn_3,
    has_flash_attn_4,
    has_ring_flash_attn,
)
from .ring import (
    RingAttentionLoadBalancerType,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from .te_attn_api import TEDotProductAttention, has_te_attn

log = logging.getLogger(__name__)


def _strip_te_extra_state(
    module: nn.Module,
    state_dict: dict[str, object],
    prefix: str,
    local_metadata: dict[str, object],
) -> None:
    del module, local_metadata
    state_dict.pop(f"{prefix}te_attn._extra_state", None)


def _restore_te_extra_state_for_load(
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
    extra_state_key = f"{prefix}te_attn._extra_state"
    if extra_state_key in state_dict:
        return
    te_attn = getattr(module, "te_attn")
    if type(te_attn).get_extra_state is not nn.Module.get_extra_state:
        state_dict[extra_state_key] = te_attn.get_extra_state()


class AttentionBackendName(StrEnum):
    """
    An enumeration of the different attention backends.
    """

    torch = "torch"
    """
    PyTorch's built-in SDPA. Works on all devices. ➡️ :class:`TorchAttentionBackend`
    """
    flash_2 = "flash_2"
    """
    Flash attention 2 from the `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ library.
    Supports Ampere (SM 8.0+) and newer NVIDIA GPUs.
    To use this with context-parallelism, `ring-flash-attn <https://github.com/zhuzilin/ring-flash-attention>`_
    is also required. ➡️ :class:`FlashAttention2Backend`
    """
    flash_3 = "flash_3"
    """
    Flash attention 3 (beta) from the `flash-attn <https://github.com/Dao-AILab/flash-attention>`_
    library ``hopper/`` subdirectory. Supports Hopper (SM 9.0) GPUs only (H100/H800).
    ➡️ :class:`FlashAttention3Backend`
    """
    flash_4 = "flash_4"
    """
    Flash attention 4, the CUTE implementation from `flash-attn <https://github.com/Dao-AILab/flash-attention>`_
    in the ``flash_attn/cute`` subdirectory. Supports SM 9.x through 11.x GPUs.
    ➡️ :class:`FlashAttention4Backend`
    """
    te = "te"
    """
    Transformer Engine attention. Supports Hopper (SM 9.0+) and newer NVIDIA GPUs.
    ➡️ :class:`TEAttentionBackend`.
    """

    def get_class(self) -> Type["AttentionBackend"]:
        if self == self.torch:
            return TorchAttentionBackend
        elif self in self.flash_2:
            return FlashAttention2Backend
        elif self == self.flash_3:
            return FlashAttention3Backend
        elif self == self.flash_4:
            return FlashAttention4Backend
        elif self == self.te:
            return TEAttentionBackend
        else:
            raise NotImplementedError(self)

    def build(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ) -> "AttentionBackend":
        return self.get_class()(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=(cache.with_namespace(f"attn_backend={self}") if cache else None),
        )

    def assert_supported(self):
        self.get_class().assert_supported()

    def assert_supports_swa(self):
        self.get_class().assert_supports_swa()

    def assert_supports_ring_cp(self):
        self.get_class().assert_supports_ring_cp()

    def assert_supports_ulysses_cp(self):
        self.get_class().assert_supports_ulysses_cp()

    def assert_supports_packed_qkv(self):
        self.get_class().assert_supports_packed_qkv()

    def assert_supports_kv_cache(self):
        self.get_class().assert_supports_kv_cache()


class AttentionBackend(nn.Module):
    """
    Encapsulates a backend for the scaled dot-product attention (SDPA) operation.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        self.assert_supported()
        if window_size != (-1, -1):
            self.assert_supports_swa()
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.scale = scale
        self.dropout_p = dropout_p
        self.window_size = window_size
        self.cache = cache
        self.cp_pg: Optional[dist.ProcessGroup] = None
        self.cp_enabled = False
        self.head_stride: int = 1

    @classmethod
    @abstractmethod
    def assert_supported(cls):
        """
        Validates that this backend is supported on the current system.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_swa(cls):
        """
        Validates that this backend supports sliding window attention (SWA).
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_ring_cp(cls):
        """
        Validates that this backend supports ring context parallelism.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_ulysses_cp(cls):
        """
        Validates that this backend supports ulysses context parallelism.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_packed_qkv(cls):
        """
        Validates that this backend supports taking QKV as a single packed tensor.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_kv_cache(cls):
        """
        Validates that this backend supports KV caching.
        Raises an error if not supported.
        """
        pass

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        """
        Warmup the buffer cache.
        """
        del max_seq_len, device

    def apply_tp(self, tp_mesh: DeviceMesh):
        """Apply tensor-parallel metadata required by the attention backend."""
        del tp_mesh

    @abstractmethod
    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        attention_sink: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        """
        Run the attention operation.
        """
        raise NotImplementedError

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """
        Apply context parallelism if supported by the backend.
        """
        if ring is not None:
            self.assert_supports_ring_cp()
        elif uly is not None:
            self.assert_supports_ulysses_cp()
        else:
            raise ValueError("One of ring or uly must be specified")

        self.cp_pg = cp_mesh.get_group()
        self.ring = ring
        self.uly = uly
        self.cp_enabled = True


class TorchAttentionBackend(AttentionBackend):
    """
    PyTorch's built-in scaled dot-product attention (SDPA) backend.
    """

    @classmethod
    def assert_supported(cls):
        pass

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ring context parallelism")

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_sliding_window_mask(
            seq_len_q=max_seq_len,
            seq_len_kv=max_seq_len,
            device=device,
            window_size=self.window_size,
        )

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        attention_sink: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        del local_k_slice

        if isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support packed QKV")

        q, k, v = qkv

        if attention_sink is not None and self.cp_enabled:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support attention sinks with CP")

        if kv_cache_manager is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support KV caching")

        attn_mask: Optional[torch.Tensor] = None
        if self.window_size != (-1, -1):
            attn_mask = self._get_sliding_window_mask(
                seq_len_q=q.shape[1],
                seq_len_kv=k.shape[1],
                device=q.device,
                window_size=self.window_size,
            )

        if any(
            opt is not None
            for opt in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
            )
        ):
            raise RuntimeError(
                f"'{self.__class__.__name__}' doesn't support intra-document masking"
            )

        if self.cp_enabled and self.uly is not None:
            assert self.cp_pg is not None
            # Transform from context-parallel to head-parallel partitioning
            # [B, T/CP, H, D] -> [B, T, H/CP, D]
            q = all_to_all_single_cp2hp(q, self.cp_pg)
            k, v = all_to_all_cp2hp([k, v], self.cp_pg)

        # NOTE: PyTorch's SDPA doesn't support GQA, so we have to do this.
        n_rep = self.n_heads // self.n_kv_heads
        # shape: (batch_size, seq_len, n_heads, head_dim)
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

        # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
        # shape: (batch_size, n_heads, seq_len, head_dim),
        #        (batch_size, n_kv_heads, seq_len, head_dim),
        #        (batch_size, n_kv_heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attention_sink is None:
            # shape: (batch_size, n_heads, seq_len, head_dim)
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                is_causal=attn_mask is None,
                scale=self.scale,
            )
        else:
            if attention_sink.numel() != q.shape[1]:
                raise RuntimeError(
                    f"attention sink has {attention_sink.numel()} heads, but query has {q.shape[1]} heads"
                )
            if attn_mask is None:
                attn_mask = self._get_sliding_window_mask(
                    seq_len_q=q.shape[-2],
                    seq_len_kv=k.shape[-2],
                    device=q.device,
                    window_size=self.window_size,
                )
            scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
            attn_weights = attn_weights.masked_fill(~attn_mask, torch.finfo(attn_weights.dtype).min)
            sinks = attention_sink.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], 1)
            combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1).float()
            probs = torch.softmax(combined, dim=-1)[..., :-1].to(q.dtype)
            probs = F.dropout(probs, p=self.dropout_p, training=self.training)
            att = torch.matmul(probs, v)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = att.transpose(1, 2)

        if self.cp_enabled and self.uly is not None:
            assert self.cp_pg is not None
            # Transform back from head-parallel to context-parallel partitioning
            # [B, T, H/CP, D] -> [B, T/CP, H, D]
            att = all_to_all_single_hp2cp(att, self.cp_pg)

        return att.contiguous()

    def _get_sliding_window_mask(
        self,
        seq_len_q: int,
        seq_len_kv: int,
        device: torch.device,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        key = f"seq_len_q={seq_len_q},seq_len_kv={seq_len_kv},window_size={window_size}"
        if self.cache is not None:
            if (mask := self.cache.get_for_device(key, device)) is not None:
                return mask

            attn_mask = self._build_sliding_window_mask(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                device=device,
                window_size=window_size,
            )
            self.cache[key] = attn_mask

            return attn_mask

        return self._build_sliding_window_mask(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            device=device,
            window_size=window_size,
        )

    @classmethod
    def _build_sliding_window_mask(
        cls,
        seq_len_q: int,
        seq_len_kv: int,
        device: torch.device,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_kv, device=device, dtype=torch.bool))

        if window_size != (-1, -1):
            sliding_window_left_mask = torch.ones_like(
                causal_mask, dtype=torch.bool, device=device
            ).triu(diagonal=-window_size[0])
            sliding_window_right_mask = torch.ones_like(
                causal_mask, dtype=torch.bool, device=device
            ).tril(diagonal=window_size[1])
            sliding_window_mask = torch.logical_and(
                sliding_window_left_mask,
                sliding_window_right_mask,
            )

            attn_mask = torch.logical_and(
                causal_mask,
                sliding_window_mask,
            )
        else:
            attn_mask = causal_mask

        return attn_mask


class FlashAttention2Backend(AttentionBackend):
    """
    SDPA from the flash-attn package. Additionally, ring-flash-attn is required for context parallelism.
    """

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_2():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the flash-attn package or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        if not has_ring_flash_attn():
            raise RuntimeError(
                f"'{cls.__name__}' requires the ring-flash-attn package for context parallelism."
            )

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        pass

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        attention_sink: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        if attention_sink is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support attention sinks")

        if isinstance(qkv, torch.Tensor):
            if kv_cache_manager is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with KV caching"
                )

            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with sliding window attention"
                )

            if self.cp_enabled:
                assert self.cp_pg is not None
                if self.ring is not None:
                    return dispatch_ring_flash_attn_qkvpacked(
                        qkv,
                        group=self.cp_pg,
                        strategy=self.ring.load_balancer,
                        cu_seqlens=cu_doc_lens,
                        max_seqlen=max_doc_len,
                        dropout_p=self.dropout_p,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.window_size,
                    )
                elif self.uly is not None:
                    # Transform packed qkv from context-parallel to head-parallel partitioning
                    # [B, T/CP, 3, H, D] -> [B, T, 3, H/CP, D]
                    qkv = all_to_all_single_cp2hp_qkvpacked(qkv, self.cp_pg)
                    B, T, _, H_local, D = qkv.shape

                    # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                    # (same on all CP ranks), so we use them directly after gathering the full sequence.

                    # Run attention with full sequence, partitioned heads
                    out = dispatch_flash_attn_qkvpacked(
                        qkv,
                        cu_seqlens=cu_doc_lens,
                        max_seqlen=max_doc_len,
                        dropout_p=self.dropout_p,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.window_size,
                    )

                    # Transform back from head-parallel to context-parallel partitioning
                    # [B, T, H/CP, D] -> [B, T/CP, H, D]
                    return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
                else:
                    raise RuntimeError("One of ring or uly must be specified")
            else:
                return dispatch_flash_attn_qkvpacked(
                    qkv,
                    cu_seqlens=cu_doc_lens,
                    max_seqlen=max_doc_len,
                    dropout_p=self.dropout_p,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.window_size,
                )

        q, k, v = qkv

        if kv_cache_manager:
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support KV caching with context parallelism"
                )

            return dispatch_flash_attn_with_kvcache(
                q,
                k=k,
                v=v,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
                k_cache=kv_cache_manager.k_cache,  # updated in-place
                v_cache=kv_cache_manager.v_cache,  # updated in-place
                cache_leftpad=kv_cache_manager.cache_leftpad,
                cache_seqlens=kv_cache_manager.cache_seqlens.expand(
                    kv_cache_manager.cache_leftpad.shape[0]
                ).contiguous(),
            )

        if self.cp_enabled:
            assert self.cp_pg is not None
            if self.ring is not None:
                return dispatch_ring_flash_attn(
                    q,
                    k,
                    v,
                    group=self.cp_pg,
                    strategy=self.ring.load_balancer,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    heads_k_stride=self.ring.head_stride,
                    local_k_slice=local_k_slice,
                    dropout_p=self.dropout_p,
                    causal=True,
                    softmax_scale=self.scale,
                    window_size=self.window_size,
                )
            elif self.uly is not None:
                # Transform from context-parallel to head-parallel partitioning
                # [B, T/CP, H, D] -> [B, T, H/CP, D]
                q = all_to_all_single_cp2hp(q, self.cp_pg)
                k, v = all_to_all_cp2hp([k, v], self.cp_pg)
                B, T, H_local, D = q.shape

                # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                # (same on all CP ranks), so we use them directly after gathering the full sequence.
                # This is the default state of cu_doc_lens and max_doc_len before a load balancer is applied.

                # Run attention with full sequence, partitioned heads
                out = dispatch_flash_attn(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    dropout_p=self.dropout_p,
                    causal=True,
                    softmax_scale=self.scale,
                    window_size=self.window_size,
                )

                # Transform back from head-parallel to context-parallel partitioning
                # [B, T, H/CP, D] -> [B, T/CP, H, D]
                return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
            else:
                raise RuntimeError("One of ring or uly must be specified")

        return dispatch_flash_attn(
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            dropout_p=self.dropout_p,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class FlashAttention3Backend(AttentionBackend):
    """
    SDPA from the flash-attn 3 package. Does not currently support context parallelism.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        if dropout_p > 0.0:
            raise RuntimeError("dropout_p > 0.0 is not supported for flash-attn 3")
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_3():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the flash-attn 3 package or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ring context parallelism")

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        pass

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        attention_sink: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        if isinstance(qkv, torch.Tensor):
            if attention_sink is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support attention sinks with packed QKV"
                )

            if kv_cache_manager is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with KV caching"
                )

            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with sliding window attention"
                )

            if self.cp_enabled:
                assert self.cp_pg is not None
                if self.ring is not None:
                    raise RuntimeError(
                        f"'{self.__class__.__name__}' doesn't support ring context parallelism"
                    )
                elif self.uly is not None:
                    # Transform packed qkv from context-parallel to head-parallel partitioning
                    # [B, T/CP, 3, H, D] -> [B, T, 3, H/CP, D]
                    qkv = all_to_all_single_cp2hp_qkvpacked(qkv, self.cp_pg)
                    B, T, _, H_local, D = qkv.shape

                    # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                    # (same on all CP ranks), so we use them directly after gathering the full sequence.

                    # Run attention with full sequence, partitioned heads
                    out = dispatch_flash_attn_3_qkvpacked(
                        qkv,
                        cu_seqlens=cu_doc_lens,
                        max_seqlen=max_doc_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.window_size,
                    )

                    # Transform back from head-parallel to context-parallel partitioning
                    # [B, T, H/CP, D] -> [B, T/CP, H, D]
                    return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
                else:
                    raise RuntimeError("One of ring or uly must be specified")

            return dispatch_flash_attn_3_qkvpacked(
                qkv,
                cu_seqlens=cu_doc_lens,
                max_seqlen=max_doc_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
            )

        q, k, v = qkv

        if attention_sink is not None and self.cp_enabled:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support attention sinks with CP")

        if kv_cache_manager:
            if attention_sink is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support attention sinks with KV caching"
                )
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support KV caching with context parallelism"
                )
            return dispatch_flash_attn_3_with_kvcache(
                q,
                k=k,
                v=v,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
                k_cache=kv_cache_manager.k_cache,  # updated in-place
                v_cache=kv_cache_manager.v_cache,  # updated in-place
                cache_leftpad=kv_cache_manager.cache_leftpad,
                cache_seqlens=kv_cache_manager.cache_seqlens.expand(
                    kv_cache_manager.cache_leftpad.shape[0]
                ).contiguous(),
            )

        if self.cp_enabled:
            if self.ring is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support ring context parallelism"
                )
            elif self.uly is not None:
                assert self.cp_pg is not None

                # Transform from context-parallel to head-parallel partitioning
                # [B, T/CP, H, D] -> [B, T, H/CP, D]
                q = all_to_all_single_cp2hp(q, self.cp_pg)
                k, v = all_to_all_cp2hp([k, v], self.cp_pg)
                B, T, H_local, D = q.shape

                # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                # (same on all CP ranks), so we use them directly after gathering the full sequence.
                # This is the default state of cu_doc_lens and max_doc_len before a load balancer is applied.

                # Run attention with full sequence, partitioned heads
                out = dispatch_flash_attn_3(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.window_size,
                )

                # Transform back from head-parallel to context-parallel partitioning
                # [B, T, H/CP, D] -> [B, T/CP, H, D]
                return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
            else:
                raise RuntimeError("One of ring or uly must be specified")

        if attention_sink is not None:
            return dispatch_flash_attn_3_with_sink(
                q,
                k,
                v,
                attention_sink,
                cu_seqlens=cu_doc_lens,
                cu_seqlens_q=cu_doc_lens_q,
                cu_seqlens_k=cu_doc_lens_k,
                max_seqlen=max_doc_len,
                max_seqlen_q=max_doc_len_q,
                max_seqlen_k=max_doc_len_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
            )

        return dispatch_flash_attn_3(
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class FlashAttention4Backend(AttentionBackend):
    """
    SDPA from flash-attn 4 (CUTE implementation).
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        if dropout_p > 0.0:
            raise RuntimeError("dropout_p > 0.0 is not supported for flash-attn 4")
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_4():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the flash-attn CUTE implementation or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ring context parallelism")

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        attention_sink: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        if attention_sink is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support attention sinks")

        assert isinstance(qkv, tuple), f"'{self.__class__.__name__}' requires unpacked QKV"
        assert local_k_slice is None, f"'{self.__class__.__name__}' doesn't support local_k_slice"

        q, k, v = qkv

        if kv_cache_manager is not None:
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support KV caching with context parallelism"
                )
            pos = int(kv_cache_manager.cache_seqlens.item())
            T_new = k.shape[1]
            kv_cache_manager.k_cache[:, pos : pos + T_new] = k
            kv_cache_manager.v_cache[:, pos : pos + T_new] = v
            seqused_k = torch.full((q.shape[0],), pos + T_new, dtype=torch.int32, device=q.device)
            return dispatch_flash_attn_4(
                q,
                kv_cache_manager.k_cache,
                kv_cache_manager.v_cache,
                seqused_k=seqused_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
            )

        if self.cp_enabled:
            if self.ring is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support ring context parallelism"
                )
            elif self.uly is not None:
                assert self.cp_pg is not None

                # Transform from context-parallel to head-parallel partitioning
                # [B, T/CP, H, D] -> [B, T, H/CP, D]
                q = all_to_all_single_cp2hp(q, self.cp_pg)
                k, v = all_to_all_cp2hp([k, v], self.cp_pg)
                B, T, H_local, D = q.shape

                # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                # (same on all CP ranks), so we use them directly after gathering the full sequence.
                # This is the default state of cu_doc_lens and max_doc_len before a load balancer is applied.

                # Run attention with full sequence, partitioned heads
                out = dispatch_flash_attn_4(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.window_size,
                )

                # Transform back from head-parallel to context-parallel partitioning
                # [B, T, H/CP, D] -> [B, T/CP, H, D]
                return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
            else:
                raise RuntimeError("One of ring or uly must be specified")

        return dispatch_flash_attn_4(
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class TEAttentionBackend(AttentionBackend):
    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )
        if not has_te_attn():
            raise RuntimeError("TransformerEngine attention is not available")
        assert TEDotProductAttention is not None
        self._te_tp_size = 1
        self._te_tp_group: Optional[dist.ProcessGroup] = None
        self._te_uses_local_shards = False
        self._te_local_shard_log_emitted = False
        self.te_attn = self._build_te_attn()
        # Transformer Engine exposes runtime metadata through ``_extra_state``.
        # It is not a model weight and is absent from HF-converted checkpoints.
        self.register_state_dict_post_hook(_strip_te_extra_state)
        self.register_load_state_dict_pre_hook(_restore_te_extra_state_for_load)

    def _build_te_attn(
        self,
        *,
        tp_size: int = 1,
        tp_group: Optional[dist.ProcessGroup] = None,
        n_heads: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
    ) -> nn.Module:
        assert TEDotProductAttention is not None
        return TEDotProductAttention(
            n_heads if n_heads is not None else self.n_heads,
            self.head_dim,
            num_gqa_groups=n_kv_heads if n_kv_heads is not None else self.n_kv_heads,
            attention_dropout=self.dropout_p,
            attn_mask_type="causal",
            window_size=(self.window_size[0], 0),  # be explicit about causal mask
            qkv_format="bshd",
            softmax_scale=self.scale,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    @classmethod
    def assert_supported(cls):
        if not has_te_attn():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the TransformerEngine package or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_cp(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        pass

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    def apply_tp(self, tp_mesh: DeviceMesh):
        tp_group = tp_mesh.get_group()
        self._te_tp_size = tp_mesh.size()
        self._te_tp_group = tp_group
        self._te_uses_local_shards = False
        self.te_attn = self._build_te_attn(tp_size=tp_mesh.size(), tp_group=tp_group)
        if self.cp_enabled:
            self._configure_context_parallel_group()

    def _te_expected_heads(self) -> Tuple[Optional[int], Optional[int]]:
        te_tp_size = getattr(self.te_attn, "tp_size", self._te_tp_size)
        te_n_heads = getattr(self.te_attn, "num_attention_heads", None)
        te_n_kv_heads = getattr(self.te_attn, "num_gqa_groups", None)
        if isinstance(te_n_heads, int) and isinstance(te_tp_size, int) and te_tp_size > 0:
            expected_q_heads = te_n_heads // te_tp_size
        else:
            expected_q_heads = None
        if hasattr(self.te_attn, "num_gqa_groups_per_partition"):
            expected_kv_heads = getattr(self.te_attn, "num_gqa_groups_per_partition")
        elif isinstance(te_n_kv_heads, int) and isinstance(te_tp_size, int) and te_tp_size > 0:
            expected_kv_heads = te_n_kv_heads // te_tp_size
        else:
            expected_kv_heads = None
        return expected_q_heads, expected_kv_heads

    def _ensure_te_matches_runtime_heads(self, q: torch.Tensor, k: torch.Tensor) -> None:
        local_q_heads = q.shape[-2]
        local_kv_heads = k.shape[-2]
        expected_q_heads, expected_kv_heads = self._te_expected_heads()

        q_matches = expected_q_heads is None or expected_q_heads == local_q_heads
        kv_matches = expected_kv_heads is None or expected_kv_heads == local_kv_heads
        if q_matches and kv_matches:
            return

        if not self._te_local_shard_log_emitted:
            log.warning(
                "Rebuilding TransformerEngine attention for local TP-sharded heads: "
                "runtime_q_heads=%s runtime_kv_heads=%s expected_q_heads=%s expected_kv_heads=%s "
                "configured_q_heads=%s configured_kv_heads=%s configured_tp_size=%s",
                local_q_heads,
                local_kv_heads,
                expected_q_heads,
                expected_kv_heads,
                self.n_heads,
                self.n_kv_heads,
                self._te_tp_size,
            )
            self._te_local_shard_log_emitted = True

        self._te_uses_local_shards = True
        self.te_attn = self._build_te_attn(
            tp_size=1,
            tp_group=None,
            n_heads=local_q_heads,
            n_kv_heads=local_kv_heads,
        )
        if self.cp_enabled:
            self._configure_context_parallel_group()

    def _configure_context_parallel_group(self):
        assert self.cp_pg is not None
        if self.ring is not None:
            if self.ring.load_balancer == RingAttentionLoadBalancerType.zig_zag:
                cp_comm_type = "p2p"
            elif self.ring.load_balancer == RingAttentionLoadBalancerType.llama3:
                cp_comm_type = "all_gather"
            else:
                raise ValueError(self.ring.load_balancer)
        elif self.uly is not None:
            cp_comm_type = "a2a"
        else:
            raise ValueError("One of ring or uly must be specified")

        self.te_attn.set_context_parallel_group(
            cp_group=self.cp_pg,
            cp_global_ranks=dist.get_process_group_ranks(self.cp_pg),
            cp_stream=torch.cuda.default_stream(),
            cp_comm_type=cp_comm_type,
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        super().apply_cp(cp_mesh, ring=ring, uly=uly)
        self._configure_context_parallel_group()

    @torch.compiler.disable(
        reason="Transformer Engine attention uses Python/pybind setup that Dynamo should not trace"
    )
    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        attention_sink: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        del local_k_slice

        if attention_sink is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support attention sinks")

        if kv_cache_manager is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support KV caching")

        if isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support packed QKV")

        q, k, v = qkv
        self._ensure_te_matches_runtime_heads(q, k)
        cu_seqlens_q = cu_doc_lens if cu_doc_lens is not None else cu_doc_lens_q
        cu_seqlens_kv = cu_doc_lens if cu_doc_lens is not None else cu_doc_lens_k
        max_seqlen_q = max_doc_len if max_doc_len is not None else max_doc_len_q
        max_seqlen_kv = max_doc_len if max_doc_len is not None else max_doc_len_k

        if cu_seqlens_q is not None or cu_seqlens_kv is not None:
            if cu_seqlens_q is None or cu_seqlens_kv is None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' requires both query and key/value cumulative "
                    "sequence lengths for packed document masking"
                )

            output_shape = q.shape
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
            out = self.te_attn(
                q,
                k,
                v,
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q,
                cu_seqlens_kv_padded=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                attn_mask_type="padding_causal",
            )
            return out.reshape(output_shape)

        return self.te_attn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
