import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.stage import (
    _normalize_model_output_as_tuple,
    flatten_args,
    map_debug_info,
)
from torch.utils._pytree import tree_map_only

from olmo_core.config import Config, DType
from olmo_core.distributed.parallel import (
    ContextParallelConfig,
    DataParallelConfig,
    ExpertParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import (
    RingAttentionLoadBalancerType,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.transformer import (
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.train_module.config import TrainModuleConfig

if TYPE_CHECKING:
    from .pipeline_train_module import TransformerPipelineTrainModule
    from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)


def _to_local_tensor(x: Any) -> Any:
    return tree_map_only(DTensor, lambda t: t.to_local(), x)


class LocalTensorPipelineStage(PipelineStage):
    """
    Keep pipeline-bound activations local because c10d P2P collectives cannot send DTensors.
    Downstream TP hooks re-wrap local tensors on the receiving stage's TP mesh.
    """

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
        save_forward_output: bool = True,
    ):
        if self.is_first:
            composite_args = args
        else:
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        composite_kwargs = kwargs or {}
        self._validate_fwd_input(args, kwargs)

        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        output_tuple = _normalize_model_output_as_tuple(output)
        local_output_tuple = tuple(_to_local_tensor(out) for out in output_tuple)

        if self.is_last and save_forward_output:
            self.output_chunks.append(output)

        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        self.fwd_cache[fwd_chunk_id] = (
            local_output_tuple,
            flat_args + flat_kwargs,
        )

        log.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(local_output_tuple),
        )
        self._validate_fwd_outputs(local_output_tuple)
        return output

    def _shape_inference(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if kwargs is None:
            kwargs = {}
        if args is None:
            raise AssertionError("Args may be an empty tuple but not None")

        if (
            self.is_first
            or self.stage_index_to_group_rank[self.stage_index - 1] == self.group_rank
        ):
            log.debug(
                "Shape inference: stage %s skipping recv, because shape info passed in via `args`",
                self.stage_index,
            )
            args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
        else:
            if len(args) != 0:
                raise AssertionError(
                    "Can't supply input args for shape inference on non-first stage"
                )
            objects = [None]
            log.debug(
                "Shape inference: stage %s receiving from stage %s",
                self.stage_index,
                self.stage_index - 1,
            )
            dist.recv_object_list(
                objects,
                src=dist.get_global_rank(
                    self.group or dist.distributed_c10d._get_default_group(),
                    self.stage_index_to_group_rank[self.stage_index - 1],
                ),
                group=self.group,
                device=self.device,
                use_batch=True,
            )
            recv_args = objects[0]
            if not isinstance(recv_args, tuple):
                raise AssertionError(f"Expected tuple, got {type(recv_args)}")
            args = recv_args

        self.inputs_meta = args
        args = tree_map_only(
            torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), args
        )

        with torch.no_grad():
            outputs = self.submod(*args, **kwargs)

        outputs_tuple = _normalize_model_output_as_tuple(outputs)
        local_outputs = tuple(_to_local_tensor(out) for out in outputs_tuple)
        outputs_meta = tuple(
            tree_map_only(torch.Tensor, lambda x: x.to("meta"), local_outputs)
        )
        log.debug(
            "Shape inference: stage %s inputs %s, outputs %s",
            self.stage_index,
            self.inputs_meta,
            outputs_meta,
        )
        self._configure_outputs_meta(outputs_meta)

        if (
            self.is_last
            or self.stage_index_to_group_rank[self.stage_index + 1] == self.group_rank
        ):
            log.debug(
                "Shape inference: stage %s skipping send to next stage",
                self.stage_index,
            )
        else:
            log.debug(
                "Shape inference: stage %s sending to stage %s",
                self.stage_index,
                self.stage_index + 1,
            )
            dist.send_object_list(
                [outputs_meta],
                dst=dist.get_global_rank(
                    self.group or dist.distributed_c10d._get_default_group(),
                    self.stage_index_to_group_rank[self.stage_index + 1],
                ),
                group=self.group,
                device=self.device,
                use_batch=True,
            )
            outputs_meta = tuple()

        return outputs_meta

    def get_bwd_send_ops(self, bwd_chunk_id: int):
        if bwd_chunk_id in self.bwd_cache:
            self.bwd_cache[bwd_chunk_id] = tuple(
                _to_local_tensor(grad) for grad in self.bwd_cache[bwd_chunk_id]
            )
        return super().get_bwd_send_ops(bwd_chunk_id)


@beta_feature
@dataclass
class TransformerPipelineParallelConfig(PipelineParallelConfig):
    """
    Transformer-specific pipeline parallel config.
    """

    split_points: Optional[List[int]] = None
    """
    A list of unique, increasing block indices that define how to split the model into stages.

    For example, ``split_points = [0, 2]`` with a 4-layer model means the model will be split into
    3 stages, with the first containing just the embedding, the second containing blocks 0 and 1,
    and the third containing blocks 2 and 3 and the language modeling head.

    If not specified the split points are determined automatically based on the schedule type.
    """

    def get_split_points(self, n_layers: int) -> List[int]:
        if self.split_points is not None:
            return self.split_points

        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified.
        num_stages_per_rank = 1 if self.schedule.is_single_stage else 2
        total_stages = self.degree * num_stages_per_rank
        num_layers = n_layers
        if total_stages > num_layers:
            raise OLMoConfigurationError("Total stages cannot be greater than the number of layers")

        base_interval = num_layers // total_stages
        extra_layers = num_layers % total_stages

        splits: List[int] = []
        current_layer = 0
        for i in range(total_stages - 1):
            if i == 0:
                current_layer += base_interval
            else:
                # Middle stages get an extra layer if there are any remaining
                if extra_layers > 0:
                    current_layer += base_interval + 1
                    extra_layers -= 1
                else:
                    current_layer += base_interval
            splits.append(current_layer)
        log.info(f"Auto generated pipeline split points will be {splits}")
        return splits

    def split_model(
        self, model: Transformer, *, pp_mesh: DeviceMesh, device: torch.device
    ) -> Tuple[List[PipelineStage], List[Transformer]]:
        split_points = self.get_split_points(model.n_layers)
        pp_rank = pp_mesh.get_local_rank()

        def build_stage(
            stage_idx: int,
            start_layer: Optional[int],
            stop_layer: Optional[int],
            is_first: bool = False,
            is_last: bool = False,
        ) -> Tuple[PipelineStage, Transformer]:
            model_chunk = copy.deepcopy(model)
            if not is_first:
                model_chunk.embeddings = None  # type: ignore

            drop_layers = start_layer is not None
            for block_idx in range(model.n_layers):
                # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
                if block_idx == start_layer:
                    drop_layers = False
                if block_idx == stop_layer:
                    drop_layers = True
                if drop_layers:
                    del model_chunk.blocks[str(block_idx)]

            if not is_last:
                model_chunk.lm_head = None  # type: ignore

            stage = LocalTensorPipelineStage(
                model_chunk,
                stage_idx,
                num_stages,
                device,
                group=pp_mesh.get_group("pp"),
            )
            return stage, model_chunk

        num_stages = len(split_points) + 1
        stage_idx = pp_rank

        stages = []
        models = []
        for stage_idx in self.stage_ids_this_rank(pp_rank, num_stages):
            start_layer = split_points[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = split_points[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = build_stage(
                stage_idx,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            log.info(
                f"PP rank {pp_rank} is building stage {stage_idx} with start layer "
                f"{start_layer}, stop layer {stop_layer}: {model_chunk}"
            )
            stages.append(stage)
            models.append(model_chunk)

        return stages, models


@dataclass
class TransformerDataParallelConfig(DataParallelConfig):
    """
    Transformer-specific data parallel config.
    """

    wrapping_strategy: TransformerDataParallelWrappingStrategy = (
        TransformerDataParallelWrappingStrategy.full
    )
    """
    The wrapping strategy.
    """

    prefetch_factor: int = 0


@dataclass
class TransformerTensorParallelConfig(TensorParallelConfig):
    """
    Transformer-specific tensor parallel config.
    """


@dataclass
class TransformerContextParallelConfig(ContextParallelConfig):
    """
    Transformer-specific context parallel config.
    """

    ring: RingContextParallelStyle | None = None
    uly: UlyssesContextParallelStyle | None = None

    def __post_init__(self):
        if self.ring is not None and self.uly is not None:
            raise NotImplementedError(
                "Only one of ring or ulysses can be specified. While not technically "
                "mutually exclusive, a combined context parallel style is not yet supported."
            )
        elif self.ring is None and self.uly is None:
            raise OLMoConfigurationError("One of ring or uly must be specified")

    @classmethod
    def zig_zag(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.zig_zag,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def llama3(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.llama3,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def ulysses(cls, degree: int) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            uly=UlyssesContextParallelStyle(),
        )


@dataclass
class TransformerExpertParallelConfig(ExpertParallelConfig):
    """
    Transformer-specific expert parallel config.
    """


@beta_feature
@dataclass
class TransformerActivationCheckpointingConfig(Config):
    """
    Defines the activation checkpointing strategy for a transformer model.
    """

    mode: TransformerActivationCheckpointingMode = TransformerActivationCheckpointingMode.full
    """
    The activation checkpointing mode.
    """

    block_interval: Optional[int] = None
    """
    Required when :data:`mode` is "selected_blocks". Determines which blocks are wrapped.
    """

    modules: Optional[List[str]] = None
    """
    Required when :data:`mode` is "selected_modules". A list of modules names to wrap for
    activation checkpointing. Globs are supported.
    """

    activation_memory_budget: Optional[float] = None
    """
    Required when :data:`mode` is "budget". Memory budget for activation checkpointing in range [0, 1].
    0 = recompute all activations, 1 = recompute none (default). Requires compilation to be enabled.

    See https://pytorch.org/blog/activation-checkpointing-techniques/ for more details.
    """

    def __post_init__(self):
        if (
            self.mode == TransformerActivationCheckpointingMode.selected_blocks
            and self.block_interval is None
        ):
            raise OLMoConfigurationError(
                "'block_interval' is required for 'selected_blocks' activation checkpointing"
            )
        elif (
            self.mode == TransformerActivationCheckpointingMode.selected_modules
            and self.modules is None
        ):
            raise OLMoConfigurationError(
                "'modules' is required for 'selected_modules' activation checkpointing"
            )


@dataclass
class TransformerTrainModuleConfig(TrainModuleConfig):
    """
    A configuration class for building :class:`TransformerTrainModule` or
    :class:`TransformerPipelineTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` and :class:`TransformerPipelineTrainModule`
        documentation for a description of the fields.
    """

    rank_microbatch_size: int
    max_sequence_length: int

    # Optimizer settings.

    optim: OptimConfig
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    pp_config: Optional[TransformerPipelineParallelConfig] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    cp_config: Optional[TransformerContextParallelConfig] = None
    ep_config: Optional[TransformerExpertParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    # Loss function settings.

    z_loss_multiplier: Optional[float] = None

    # Checkpoint settings.

    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    # Other train settings.

    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> Union["TransformerTrainModule", "TransformerPipelineTrainModule"]:
        """
        Build the corresponding :class:`TransformerTrainModule` or :class:`TransformerPipelineTrainModule.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.
        """
        from .pipeline_train_module import TransformerPipelineTrainModule
        from .train_module import TransformerTrainModule

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        if self.pp_config is not None:
            return TransformerPipelineTrainModule(
                model=model,
                device=device,
                **kwargs,
            )
        else:
            return TransformerTrainModule(
                model=model,
                device=device,
                **kwargs,
            )


@beta_feature
@dataclass
class TransformerPipelineTrainModuleConfig(TransformerTrainModuleConfig):
    """
    Kept for backwards compatibility, but please use :class:`TransformerTrainModuleConfig` instead.
    """

    def __post_init__(self):
        if self.pp_config is None:
            raise OLMoConfigurationError("'pp_config' is required")
