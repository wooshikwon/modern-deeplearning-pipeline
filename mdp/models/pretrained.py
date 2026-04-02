"""PretrainedResolver — URI 기반 사전학습 모델 로딩."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)

_PROTOCOLS = ("hf://", "timm://", "ultralytics://", "local://")


class PretrainedResolver:
    """URI 패턴으로 사전학습 모델을 로드한다.

    지원 프로토콜::

        hf://bert-base-uncased              → AutoModel.from_pretrained
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
        protocol, identifier = cls._parse_uri(uri)

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
        """HuggingFace 모델 로딩."""
        if class_path is not None:
            module_path, _, cls_name = class_path.rpartition(".")
            mod = importlib.import_module(module_path)
            model_cls = getattr(mod, cls_name)
        else:
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise ImportError(
                    "transformers 패키지가 필요합니다: pip install transformers"
                ) from e
            model_cls = AutoModel

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
