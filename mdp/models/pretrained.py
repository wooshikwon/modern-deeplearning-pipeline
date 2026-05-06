"""PretrainedResolver — URI 기반 사전학습 모델 로딩."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_PROTOCOLS = ("hf://", "timm://", "ultralytics://", "local://")


@dataclass(frozen=True)
class PretrainedLoadSpec:
    """Pretrained 모델 로딩 옵션을 정규화한 plan."""

    uri: str
    protocol: str
    torch_dtype: torch.dtype | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    device_map: str | None = None

    @classmethod
    def from_options(
        cls,
        uri: str,
        *,
        dtype: str | torch.dtype | None = None,
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        device_map: str | None = None,
    ) -> "PretrainedLoadSpec":
        """CLI/Recipe option 이름을 받아 로더용 타입으로 정규화한다."""
        protocol, _ = PretrainedResolver._parse_uri(uri)
        dtype_value = torch_dtype if torch_dtype is not None else dtype
        normalized_dtype = cls._normalize_dtype(dtype_value)
        spec = cls(
            uri=uri,
            protocol=protocol,
            torch_dtype=normalized_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        spec.validate()
        return spec

    @staticmethod
    def _normalize_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        if not isinstance(dtype, str):
            raise TypeError(f"dtype은 str 또는 torch.dtype이어야 합니다: {type(dtype).__name__}")
        try:
            value = getattr(torch, dtype)
        except AttributeError as e:
            raise ValueError(f"지원하지 않는 torch dtype: {dtype!r}") from e
        if not isinstance(value, torch.dtype):
            raise ValueError(f"지원하지 않는 torch dtype: {dtype!r}")
        return value

    def validate(self) -> None:
        if self.protocol == "hf":
            return
        invalid: list[str] = []
        if self.trust_remote_code:
            invalid.append("trust_remote_code")
        if self.attn_implementation is not None:
            invalid.append("attn_implementation")
        if self.device_map is not None:
            invalid.append("device_map")
        if self.protocol in {"ultralytics", "local"} and self.torch_dtype is not None:
            invalid.append("torch_dtype")
        if invalid:
            names = ", ".join(invalid)
            raise ValueError(
                f"{self.protocol}:// pretrained 로더에는 {names} 옵션을 전달할 수 없습니다."
            )

    def to_loader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        return kwargs


class PretrainedResolver:
    """URI 패턴으로 사전학습 모델을 로드한다.

    지원 프로토콜::

        hf://bert-base-uncased              → config.architectures[0] 클래스로 로드
        hf://bert-base-uncased?class_path=... → 동적 임포트 후 .from_pretrained
        timm://resnet50                      → timm.create_model(pretrained=True)
        ultralytics://yolov8n.pt             → YOLO(identifier)
        local:///path/to/model.pt            → torch.load
        bert-base-uncased                    → hf://로 취급
    """

    @classmethod
    def load(
        cls,
        uri: str,
        class_path: str | None = None,
        load_spec: PretrainedLoadSpec | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """URI에서 사전학습 모델을 로드한다.

        Args:
            uri: 프로토콜://식별자 형식의 URI.
            class_path: HF 모델의 커스텀 클래스 경로 (hf:// 전용).
            **kwargs: 모델 로딩에 전달할 추가 인자.

        Returns:
            로드된 nn.Module.
        """
        spec = load_spec or PretrainedLoadSpec.from_options(uri)
        if spec.uri != uri:
            raise ValueError("load_spec.uri와 uri가 일치해야 합니다.")
        protocol, identifier = cls._parse_uri(uri)
        kwargs = {**spec.to_loader_kwargs(), **kwargs}

        handlers = {
            "hf": cls._load_hf,
            "timm": cls._load_timm,
            "ultralytics": cls._load_ultralytics,
            "local": cls._load_local,
        }

        handler = handlers.get(protocol)
        if handler is None:
            raise ValueError(f"지원하지 않는 프로토콜: {protocol!r}")

        if protocol == "hf":
            return handler(identifier, class_path=class_path, **kwargs)
        return handler(identifier, **kwargs)

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        """URI를 (프로토콜, 식별자)로 분리한다."""
        for prefix in _PROTOCOLS:
            if uri.startswith(prefix):
                protocol = prefix.rstrip(":/")
                identifier = uri[len(prefix):]
                return protocol, identifier

        # 접두사 없으면 hf://로 취급
        return "hf", uri

    @staticmethod
    def _load_hf(
        identifier: str,
        class_path: str | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """HuggingFace 모델 로딩.

        class_path가 명시되어 있으면 그대로 사용 (Recipe 기반 경로).
        class_path가 None이면 config.architectures[0]에서 직접 모델 클래스를 결정한다.
        architectures가 없거나 빈 리스트이면 에러를 발생시킨다 (폴백 없음).
        """
        try:
            import transformers
        except ImportError as e:
            raise ImportError(
                "transformers 패키지가 필요합니다: pip install transformers"
            ) from e

        if class_path is not None:
            module_path, _, cls_name = class_path.rpartition(".")
            mod = importlib.import_module(module_path)
            model_cls = getattr(mod, cls_name)
        else:
            config = transformers.AutoConfig.from_pretrained(identifier, **kwargs)
            architectures = getattr(config, "architectures", None) or []
            if not architectures:
                raise ValueError(
                    f"모델 '{identifier}'의 config에 architectures 필드가 없습니다. "
                    "HuggingFace Hub에서 올바른 모델 식별자를 확인하세요."
                )
            cls_name = architectures[0]
            model_cls = getattr(transformers, cls_name, None)
            if model_cls is None:
                raise ValueError(
                    f"transformers에 '{cls_name}' 클래스가 없습니다."
                )
            logger.info(
                "config.architectures=%s → %s", cls_name, model_cls.__name__,
            )

        logger.info("HF 모델 로딩: %s (class=%s)", identifier, model_cls.__name__)
        return model_cls.from_pretrained(identifier, **kwargs)

    @staticmethod
    def _load_timm(identifier: str, **kwargs: Any) -> nn.Module:
        """timm 모델 로딩."""
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm 패키지가 필요합니다: pip install timm"
            ) from e

        logger.info("timm 모델 로딩: %s", identifier)
        return timm.create_model(identifier, pretrained=True, **kwargs)

    @staticmethod
    def _load_ultralytics(identifier: str, **kwargs: Any) -> nn.Module:
        """Ultralytics YOLO 모델 로딩."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics 패키지가 필요합니다: pip install ultralytics"
            ) from e

        logger.info("Ultralytics 모델 로딩: %s", identifier)
        return YOLO(identifier, **kwargs)

    @staticmethod
    def _load_local(path: str, **kwargs: Any) -> nn.Module:
        """로컬 파일에서 모델 로딩."""
        import torch

        logger.warning(
            "local:// URI는 weights_only=False로 pickle 코드를 실행합니다. "
            "신뢰할 수 없는 소스의 .pt 파일은 사용하지 마세요: %s", path,
        )
        return torch.load(path, map_location="cpu", weights_only=False, **kwargs)
