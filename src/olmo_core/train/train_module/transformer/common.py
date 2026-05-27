import logging
from typing import List, Optional, TypeVar, cast

import torch
from torch.distributed import DeviceMesh
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel.style import PrepareModuleInput

from olmo_core.distributed.parallel import (
    DataParallelType,
    get_cp_mesh,
    get_device_mesh_info,
    get_dp_model_mesh,
    get_ep_mesh,
    get_pp_mesh,
    get_tp_mesh,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import MoETransformer, Transformer

from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)


M = TypeVar("M", Transformer, List[Transformer])


def _mesh_group_names(mesh: DeviceMesh) -> tuple[str, ...]:
    return tuple(str(name) for name in getattr(mesh, "_dim_group_names", ()))


def _patch_prepare_module_input_for_pipeline_tp() -> None:
    """Re-wrap DTensor inputs that arrive from a different PP stage's TP mesh."""
    if getattr(PrepareModuleInput, "_olmo_core_pp_tp_patch", False):
        return

    original_prepare_input_arg = PrepareModuleInput._prepare_input_arg

    def patched_prepare_input_arg(self, input, mesh, input_layout, desired_layout):
        if input_layout is not None and isinstance(input, DTensor):
            input_mesh = input.device_mesh
            if _mesh_group_names(input_mesh) != _mesh_group_names(mesh):
                input = input.to_local()
        return original_prepare_input_arg(self, input, mesh, input_layout, desired_layout)

    PrepareModuleInput._prepare_input_arg = patched_prepare_input_arg
    PrepareModuleInput._olmo_core_pp_tp_patch = True


def _retain_process_group_ref(refs: list, name: str, group: object) -> None:
    refs.append((name, group))
    group_name = getattr(group, "group_name", None)
    if not group_name:
        return
    try:
        c10d._register_process_group(group_name, group)
    except Exception:
        log.debug("Process group %s was already registered or could not be re-registered", group_name)


def _retain_mesh_refs(module: object, name: str, mesh: DeviceMesh) -> None:
    """Keep process groups alive for DTensor hooks that store group names."""
    refs = getattr(module, "_olmo_core_parallel_refs", None)
    if refs is None:
        refs = []
        setattr(module, "_olmo_core_parallel_refs", refs)
    refs.append((name, mesh))

    candidate_meshes = [mesh]
    try:
        root_mesh = mesh._get_root_mesh()
        if root_mesh is not mesh:
            candidate_meshes.append(root_mesh)
            refs.append((f"{name}_root", root_mesh))
        candidate_meshes.extend(root_mesh._flatten_mapping.values())
    except Exception:
        pass

    seen_meshes: set[int] = set()
    for idx, candidate in enumerate(candidate_meshes):
        if id(candidate) in seen_meshes:
            continue
        seen_meshes.add(id(candidate))
        try:
            groups = [candidate.get_group()] if candidate.ndim == 1 else candidate.get_all_groups()
        except Exception:
            continue
        for group_idx, group in enumerate(groups):
            group_name = getattr(group, "group_name", str(group_idx))
            _retain_process_group_ref(refs, f"{name}_mesh_{idx}_group_{group_name}", group)
        try:
            refs.append((f"{name}_mesh_{idx}_pg_registry", tuple(candidate._pg_registry.items())))
        except Exception:
            pass


def parallelize_model(
    model: M,
    *,
    world_mesh: Optional[DeviceMesh],
    device: torch.device,
    max_sequence_length: Optional[int] = None,
    rank_microbatch_size: Optional[int] = None,
    compile_model: bool = False,
    float8_config: Optional[Float8Config] = None,
    dp_config: Optional[TransformerDataParallelConfig] = None,
    tp_config: Optional[TransformerTensorParallelConfig] = None,
    cp_config: Optional[TransformerContextParallelConfig] = None,
    ep_config: Optional[TransformerExpertParallelConfig] = None,
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
    pp_enabled: bool = False,
) -> M:
    _patch_prepare_module_input_for_pipeline_tp()
    model_parts: List[Transformer] = [model] if isinstance(model, Transformer) else model

    pp_mesh: Optional[DeviceMesh] = None
    if pp_enabled:
        assert world_mesh is not None
        pp_mesh = get_pp_mesh(world_mesh)
        for m in model_parts:
            _retain_mesh_refs(m, "world_mesh", world_mesh)
            _retain_mesh_refs(m, "pp_mesh", pp_mesh)
            m.apply_pp(pp_mesh)

    # Maybe apply FP8 training.
    if float8_config is not None and float8_config.enabled:
        for m in model_parts:
            m.apply_fp8(float8_config)
            log.info("Swapped linear layers to Float8 linear layers\n%s", m)

    # Maybe apply context parallelism.
    if cp_config is not None:
        assert world_mesh is not None
        cp_mesh = get_cp_mesh(world_mesh)
        for m in model_parts:
            _retain_mesh_refs(m, "cp_mesh", cp_mesh)
            m.apply_cp(cp_mesh, ring=cp_config.ring, uly=cp_config.uly)
        log.info(f"Applied context parallelism to the model with {get_device_mesh_info(cp_mesh)}")

    # Maybe apply tensor.
    if tp_config is not None:
        if ep_config is not None:
            raise NotImplementedError("TP + EP is not implemented yet")
        assert world_mesh is not None
        tp_mesh = get_tp_mesh(world_mesh)
        for m in model_parts:
            _retain_mesh_refs(m, "tp_mesh", tp_mesh)
            m.apply_tp(tp_mesh)
        tp_config.maybe_enable_async_tp(tp_mesh)
        log.info(f"Applied tensor parallelism to the model with {get_device_mesh_info(tp_mesh)}")

    # Maybe apply expert parallelism.
    if ep_config is not None:
        assert world_mesh is not None
        ep_mesh = get_ep_mesh(world_mesh)
        for m in model_parts:
            _retain_mesh_refs(m, "ep_mesh", ep_mesh)
            if not m.is_moe:
                raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
            cast(MoETransformer, m).apply_ep(ep_mesh)
        log.info(f"Applied expert parallelism to the model with {get_device_mesh_info(ep_mesh)}")

    # Maybe apply activation checkpointing.
    if ac_config is not None:
        for m in model_parts:
            m.apply_activation_checkpointing(
                ac_config.mode,
                block_interval=ac_config.block_interval,
                modules=ac_config.modules,
                activation_memory_budget=ac_config.activation_memory_budget,
            )
        log.info(f"Applied '{ac_config.mode}' activation checkpointing to the model")

    # Maybe compile.
    if compile_model:
        if torch.cuda.is_available():
            for m in model_parts:
                m.apply_compile()
            log.info("Applied torch.compile() to the model")
        else:
            log.warning("Skipping model compilation since CUDA is not available")

    # Maybe shard/replicate according to data parallel config.
    if dp_config is not None:
        assert world_mesh is not None
        dp_mesh = get_dp_model_mesh(world_mesh)
        param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
        for m in model_parts:
            _retain_mesh_refs(m, "dp_mesh", dp_mesh)
        if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
            for m in model_parts:
                if m.is_moe:
                    cast(MoETransformer, m).prepare_experts_for_fsdp(
                        world_mesh,
                        param_dtype=param_dtype,
                        reduce_dtype=dp_config.reduce_dtype.as_pt(),
                        pp_enabled=pp_enabled,
                    )
                m.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                    wrapping_strategy=dp_config.wrapping_strategy,
                    pp_enabled=pp_enabled,
                    prefetch_factor=dp_config.prefetch_factor,
                )
            log.info(f"Applied FSDP to the model with {get_device_mesh_info(dp_mesh)}")
        elif dp_config.name == DataParallelType.ddp:
            for m in model_parts:
                if m.is_moe:
                    cast(MoETransformer, m).prepare_experts_for_ddp(world_mesh)
                m.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")
        else:
            raise NotImplementedError(dp_config.name)

    # Materialize and init parameters.
    log.info("Initializing model weights...")
    for model_part_idx, m in enumerate(model_parts):
        m.init_weights(
            max_seq_len=max_sequence_length,
            max_local_microbatch_size=rank_microbatch_size,
            device=device,
            world_mesh=world_mesh,
            model_part_idx=model_part_idx,
        )

    return model
