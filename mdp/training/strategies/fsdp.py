"""FullyShardedDataParallel (FSDP) training strategy."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import torch
from torch import nn

from mdp.training.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class FSDPStrategy(BaseStrategy):
    """Wraps a model with :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.

    Parameters
    ----------
    sharding_strategy:
        One of ``"FULL_SHARD"``, ``"SHARD_GRAD_OP"``, ``"NO_SHARD"``,
        or ``"HYBRID_SHARD"``.  Defaults to ``"FULL_SHARD"``.
    mixed_precision:
        When *True*, enables mixed-precision policy matching the given *precision*.
    precision:
        ``"bf16"`` or ``"fp16"``.  Determines the MixedPrecision dtype.
    min_num_params:
        Minimum parameter count for auto-wrapping.  Layers with at least
        this many parameters become individual FSDP units.
    """

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = True,
        backend: str = "nccl",
        cpu_offload: bool = False,
        precision: str = "bf16",
        min_num_params: int = 1_000_000,
        auto_wrap_cls: str | list[str] | None = None,
    ) -> None:
        self.sharding_strategy_name = sharding_strategy
        self.mixed_precision = mixed_precision
        self.backend = backend
        self.cpu_offload = cpu_offload
        self.precision = precision
        self.min_num_params = min_num_params
        self.auto_wrap_cls = auto_wrap_cls
        self._local_rank: int | None = None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> nn.Module:  # noqa: ARG002
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        if device.type == "cuda":
            target_device = torch.device(f"cuda:{self._local_rank}")
        else:
            target_device = device

        sharding = getattr(ShardingStrategy, self.sharding_strategy_name)

        # Auto-wrap policy: 4단계 우선순위
        # 1) 사용자 명시 auto_wrap_cls (escape hatch, 최우선)
        # 2) MDP 계약: BaseModel._block_classes
        # 3) HF 호환: PreTrainedModel._no_split_modules
        # 4) size 기반 (LoRA+FSDP 조합에서는 에러)
        auto_wrap_policy = self._resolve_wrap_policy(model, size_based_auto_wrap_policy)

        fsdp_kwargs: dict[str, Any] = {
            "sharding_strategy": sharding,
            "auto_wrap_policy": auto_wrap_policy,
            # LoRA+FSDP에서 mixed requires_grad(frozen base + trainable adapter)를 허용.
            # PyTorch 2.0+에서 원본 파라미터 객체를 유지하여 FlatParameter 균일성 제약 제거.
            "use_orig_params": True,
        }

        if device.type == "cuda":
            fsdp_kwargs["device_id"] = target_device

        if self.mixed_precision and device.type == "cuda":
            dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
                cast_forward_inputs=True,
            )
            # LoRA 어댑터는 float32로 초기화되어 base 모델 dtype과 충돌.
            # FSDP FlatParameter 생성 전에 전체를 target dtype으로 통일한다.
            model = model.to(dtype)

        if self.cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        return FSDP(model, **fsdp_kwargs)

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )
        from safetensors.torch import save_file

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(state_dict, path)

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )
        from safetensors.torch import load_file

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
        return model

    def setup_models(
        self, models: dict[str, nn.Module], device: torch.device,
        trainable_names: set[str] | None = None,
        optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ) -> dict[str, nn.Module]:
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        trainable_names = trainable_names or set()
        wrapped = {}
        for name, model in models.items():
            if name in trainable_names:
                wrapped[name] = self.setup(model, device)
            else:
                # frozen → NO_SHARD (forward only, no gradient communication)
                if device.type == "cuda":
                    model = model.to(torch.device(f"cuda:{self._local_rank}"))
                wrapped[name] = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD)
        return wrapped

    def cleanup(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def _resolve_wrap_policy(
        self,
        model: nn.Module,
        size_based_auto_wrap_policy: Any,
    ) -> Any:
        """Wrap policy를 4단계 우선순위로 결정한다.

        1. 사용자 명시 ``auto_wrap_cls`` — escape hatch, 최우선
        2. MDP 계약 — ``BaseModel._block_classes`` 가 non-None 이면
           ``transformer_auto_wrap_policy`` 로 해당 클래스들을 wrap unit 으로 사용
        3. HF 호환 폴백 — ``_no_split_modules`` 가 있으면 그것을 사용
        4. size 기반 폴백 — 위 모두 없을 때.
           단, LoRA(PEFT)+FSDP 조합이면 size 기반은 위험하므로 즉시 에러
        """
        block_classes: set[str] | None = None
        source: str = ""

        # 1) 사용자 명시
        if self.auto_wrap_cls is not None:
            if isinstance(self.auto_wrap_cls, str):
                block_classes = {self.auto_wrap_cls}
            else:
                block_classes = set(self.auto_wrap_cls)
            source = "auto_wrap_cls (user override)"

        # 2) MDP 계약: BaseModel._block_classes
        elif getattr(model, "_block_classes", None):
            block_classes = set(model._block_classes)
            source = "_block_classes"

        # 3) HF 호환: _no_split_modules
        # _no_split_modules는 단축명("LlamaDecoderLayer")만 제공하므로
        # named_modules()를 스캔하여 전체 경로로 resolve한다.
        elif getattr(model, "_no_split_modules", None):
            short_names = set(model._no_split_modules)
            source = "_no_split_modules (HF compat)"
            full_paths: set[str] = set()
            for short_name in short_names:
                for _, mod in model.named_modules():
                    if type(mod).__name__ == short_name:
                        cls = type(mod)
                        full_paths.add(f"{cls.__module__}.{cls.__qualname__}")
                        break
            block_classes = full_paths if full_paths else short_names

        if block_classes:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            layer_classes = {
                self._resolve_layer_class(name) for name in block_classes
            }
            logger.info(
                "FSDP wrap policy: transformer_auto_wrap_policy "
                "(source=%s, classes=%s)",
                source,
                block_classes,
            )
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=layer_classes,
            )

        # 4) 위험 조합 차단: LoRA(PEFT) + FSDP + block 미선언 → 즉시 에러
        if hasattr(model, "peft_config"):
            raise ValueError(
                "LoRA(PEFT) + FSDP 조합에는 모델의 반복 블록 선언이 필수입니다.\n"
                "해결 방법:\n"
                "  1. BaseModel을 사용하여 _block_classes를 선언하세요.\n"
                "  2. 또는 HF PreTrainedModel을 직접 사용하세요 "
                "(_no_split_modules가 자동 제공됩니다).\n"
                "  3. timm 모델이라면 BaseModel로 감싸서 "
                "_block_classes를 선언하세요."
            )

        # non-LoRA FSDP: size_based는 안전 (heterogeneous requires_grad 없음)
        logger.info(
            "FSDP wrap policy: size_based_auto_wrap_policy "
            "(min_num_params=%d)",
            self.min_num_params,
        )
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self.min_num_params,
        )

    @staticmethod
    def _resolve_layer_class(cls_name: str) -> type:
        """클래스 이름 또는 전체 경로에서 transformer layer 클래스를 resolve한다."""
        import importlib

        if "." in cls_name:
            module_path, _, class_name = cls_name.rpartition(".")
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        # 단축명: transformers 패키지에서 탐색
        try:
            import transformers
            cls = getattr(transformers, cls_name, None)
            if cls is not None:
                return cls
        except ImportError:
            pass
        raise ValueError(
            f"transformer layer class '{cls_name}'를 찾을 수 없습니다. "
            "전체 경로를 사용하세요."
        )
