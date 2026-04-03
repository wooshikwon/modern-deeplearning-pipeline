"""MDP 서빙 서버 — recipe 기반 동적 API 엔드포인트."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_app(handler: Any, recipe: Any) -> Any:
    """recipe 기반으로 FastAPI 앱을 생성한다."""
    from fastapi import FastAPI, Request

    @asynccontextmanager
    async def lifespan(app):
        if hasattr(handler, "start"):
            await handler.start()
        yield
        if hasattr(handler, "stop"):
            await handler.stop()

    app = FastAPI(title=f"MDP: {recipe.name}", lifespan=lifespan)

    @app.post("/predict")
    async def predict(request: Request):
        raw = await _parse_request(request, recipe.data.fields, recipe.task)
        result = await handler.handle(raw)
        if hasattr(result, "body"):
            return result  # StreamingResponse
        return result

    @app.get("/health")
    def health():
        return {"status": "ok", "model": recipe.name, "task": recipe.task}

    return app


async def _parse_request(request: Any, fields: dict, task: str) -> dict:
    """fields와 task에 따라 HTTP 요청을 파싱한다."""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        result = {}
        for role, col in (fields or {}).items():
            if role == "image" and col in form:
                upload = form[col]
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(await upload.read())).convert("RGB")
                result["image"] = image
        return result

    body = await request.json()
    return body


def create_handler(
    model: Any, recipe: Any, model_dir: Path | None = None, serving_config: Any | None = None,
) -> Any:
    """model + recipe에서 task에 맞는 handler를 생성한다.

    Args:
        model: eval 모드의 모델. 호출자가 reconstruct + merge를 완료한 상태.
        recipe: Settings.recipe 객체.
        model_dir: tokenizer 파일이 있는 디렉토리. None이면 recipe에서 fallback.
        serving_config: ServingConfig 객체. PredictHandler에 배치 설정을 전달한다.
    """
    from mdp.serving.handlers import GenerateHandler, PredictHandler

    tokenizer = _load_tokenizer(model_dir, recipe)
    transform = _load_transform(recipe)

    if recipe.task in ("text_generation", "seq2seq"):
        return GenerateHandler(model, tokenizer, recipe)
    else:
        return PredictHandler(model, tokenizer, transform, recipe, serving_config=serving_config)


def _load_tokenizer(model_dir: Path | None, recipe: Any) -> Any:
    """tokenizer를 로드한다. model_dir 우선, 없으면 recipe fallback."""
    # 1. model_dir에서 시도
    if model_dir is not None:
        tokenizer_json = model_dir / "tokenizer.json"
        tokenizer_config = model_dir / "tokenizer_config.json"
        if tokenizer_json.exists() or tokenizer_config.exists():
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(model_dir))

    # 2. recipe에서 fallback
    if recipe.data.tokenizer:
        pretrained = (
            recipe.data.tokenizer.get("pretrained")
            if isinstance(recipe.data.tokenizer, dict)
            else getattr(recipe.data.tokenizer, "pretrained", None)
        )
        if pretrained:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(pretrained)

    return None


def _load_transform(recipe: Any) -> Any:
    """recipe에서 val transform을 로드한다."""
    if recipe.data.augmentation:
        from mdp.data.transforms import build_transforms
        aug = recipe.data.augmentation
        val_config = aug.get("val") if isinstance(aug, dict) else getattr(aug, "val", None)
        if val_config:
            return build_transforms(val_config)
    return None
