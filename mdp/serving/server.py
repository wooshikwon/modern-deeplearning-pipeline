"""MDP 서빙 서버 — recipe 기반 동적 API 엔드포인트."""

from __future__ import annotations

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
        fields = recipe.data.dataset.get("fields") if isinstance(recipe.data.dataset, dict) else None
        raw = await _parse_request(request, fields, recipe.task)
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
            if role == "image":
                from PIL import Image
                import io
                upload = form[role] if role in form else form.get(col)
                if upload is None:
                    continue
                image = Image.open(io.BytesIO(await upload.read())).convert("RGB")
                result["image"] = image
            else:
                upload = form[role] if role in form else form.get(col)
                if upload is None:
                    continue
                result[role] = upload if isinstance(upload, str) else (await upload.read()).decode("utf-8")
        return result

    body = await request.json()
    if not fields:
        return body
    result = dict(body)
    for role, col in fields.items():
        if role in result:
            continue
        if col in body:
            result[role] = body[col]
    return result


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

    tokenizer = _load_tokenizer(model_dir, recipe, model)
    transform = _load_transform(recipe)

    if recipe.task in ("text_generation", "seq2seq"):
        if tokenizer is None:
            raise ValueError(
                f"'{recipe.task}' 태스크에는 tokenizer가 필수입니다. "
                "model_dir에 tokenizer 파일이 있거나 recipe.data.collator에 tokenizer를 설정하세요."
            )
        return GenerateHandler(model, tokenizer, recipe)
    else:
        return PredictHandler(model, tokenizer, transform, recipe, serving_config=serving_config)


def _load_tokenizer(model_dir: Path | None, recipe: Any, model: Any = None) -> Any:
    """tokenizer를 로드한다. model_dir 우선, 없으면 recipe fallback.

    model이 주어지면 아키텍처에 맞는 padding_side를 자동 설정한다.
    decoder-only(LLaMA 등) → 'left', encoder-decoder(T5, BART 등) → 'right'.
    """
    from mdp.serving.model_loader import _resolve_padding_side
    padding_side = _resolve_padding_side(model) if model is not None else "left"

    # 1. model_dir에서 시도
    if model_dir is not None:
        tokenizer_json = model_dir / "tokenizer.json"
        tokenizer_config = model_dir / "tokenizer_config.json"
        if tokenizer_json.exists() or tokenizer_config.exists():
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(model_dir), padding_side=padding_side)

    # 2. recipe에서 fallback — tokenizer는 collator _component_의 init_args
    tokenizer_name = recipe.data.collator.get("tokenizer") if isinstance(recipe.data.collator, dict) else None
    if tokenizer_name:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name, padding_side=padding_side)

    return None


def _load_transform(recipe: Any) -> Any:
    """recipe에서 val transform을 로드한다.

    augmentation은 Dataset _component_의 init_args에 있다.
    val_dataset이 있으면 그쪽의 augmentation을, 없으면 dataset의 것을 사용한다.
    """
    # val_dataset 우선, 없으면 dataset
    ds_config = recipe.data.val_dataset or recipe.data.dataset
    if not isinstance(ds_config, dict):
        return None
    augmentation = ds_config.get("augmentation")
    if augmentation:
        from mdp.data.transforms import build_transforms
        return build_transforms(augmentation)
    return None
