"""서빙 핸들러 — task별 요청 처리 전략."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable

import torch
from starlette.responses import StreamingResponse

from mdp.models.forward import make_forward_fn

logger = logging.getLogger(__name__)


_GENERATION_KEYS = (
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
    "num_beams",
    "repetition_penalty",
)

_GENERATION_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "do_sample": True,
    "num_beams": 1,
    "repetition_penalty": 1.0,
}


def resolve_generation_kwargs(
    recipe_generation: Any | None,
    cli_args_or_serving_args: Any | None = None,
) -> dict[str, Any]:
    """Resolve generation kwargs with defaults < recipe < explicit args."""
    resolved = dict(_GENERATION_DEFAULTS)

    recipe_values = _generation_values(recipe_generation)
    for key in _GENERATION_KEYS:
        value = recipe_values.get(key)
        if value is not None:
            resolved[key] = value

    explicit_values = _generation_values(cli_args_or_serving_args)
    for key in _GENERATION_KEYS:
        value = explicit_values.get(key)
        if value is not None:
            resolved[key] = value

    return resolved


def _generation_values(source: Any | None) -> dict[str, Any]:
    if source is None:
        return {}
    if hasattr(source, "model_dump"):
        return source.model_dump(exclude_none=True)
    if isinstance(source, dict):
        return source
    return {
        key: getattr(source, key)
        for key in _GENERATION_KEYS
        if hasattr(source, key)
    }


class GenerateHandler:
    """text_generation/seq2seq용. autoregressive token streaming via SSE."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        recipe: Any,
    ) -> None:
        from transformers import TextIteratorStreamer

        self.model = model
        self.tokenizer = tokenizer
        self._streamer_cls = TextIteratorStreamer
        self._device = next(model.parameters()).device
        self.generation_config = recipe.generation or {}
        if hasattr(self.generation_config, "model_dump"):
            self.generation_config = self.generation_config.model_dump(exclude_none=True)
        self._lock = asyncio.Lock()

    async def handle(self, raw_input: dict) -> StreamingResponse:
        # locked()로 atomic 503 — race window 제거
        if self._lock.locked():
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "concurrent generation not allowed"}, status_code=503)
        await self._lock.acquire()

        # lock 획득 성공 — 이후 모든 해제는 generator finally에서
        text = raw_input.get("text", "")
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        streamer = self._streamer_cls(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
        }
        gen_kwargs.update(resolve_generation_kwargs(self.generation_config, raw_input))

        # thread wrapper: 예외 캡처 + sentinel 주입으로 deadlock 방지
        thread_exc: list[BaseException | None] = [None]

        def _generate():
            try:
                self.model.generate(**gen_kwargs)
            except BaseException as e:
                thread_exc[0] = e
                # sentinel 주입 — streamer iterator 탈출
                q = getattr(streamer, "text_queue", getattr(streamer, "queue", None))
                if q is not None:
                    q.put(streamer.stop_signal)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        # generator가 lock + thread 라이프사이클을 소유
        async def event_stream():
            try:
                for token_text in streamer:
                    if thread_exc[0] is not None:
                        break
                    yield f"data: {json.dumps({'token': token_text})}\n\n"
                if thread_exc[0] is not None:
                    yield f"data: {json.dumps({'error': str(thread_exc[0])})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                thread.join(timeout=30)
                self._lock.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")


class PredictHandler:
    """classification/detection용. max_batch_size > 1이면 BatchScheduler, 아니면 즉시 처리."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any | None,
        transform: Callable | None,
        recipe: Any,
        serving_config: Any | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.task = recipe.task
        self._device = next(model.parameters()).device
        self.forward_fn = make_forward_fn(model)
        max_bs = 1
        window = 50.0
        if serving_config is not None:
            max_bs = getattr(serving_config, "max_batch_size", 1)
            window = getattr(serving_config, "batch_window_ms", 50.0)
        self._use_batching = max_bs > 1
        if self._use_batching:
            self.scheduler = _BatchScheduler(
                model,
                forward_fn=self.forward_fn,
                max_batch_size=max_bs,
                batch_window_ms=window,
            )

    async def handle(self, raw_input: dict) -> dict:
        preprocessed = await asyncio.to_thread(self._preprocess, raw_input)
        if self._use_batching:
            return await self.scheduler.submit(preprocessed)
        return await self._infer_single(preprocessed)

    async def _infer_single(self, preprocessed: dict) -> dict:
        """배치 없이 단일 요청을 즉시 처리한다."""
        inputs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in preprocessed.items()
        }
        outputs = await asyncio.to_thread(self._run_model, inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = {"logits": outputs}
        return _BatchScheduler._unbatch(outputs, 0)

    def _run_model(self, inputs: dict) -> Any:
        with torch.no_grad():
            return self.forward_fn(inputs)

    def _preprocess(self, raw_input: dict) -> dict:
        """raw 입력을 모델 입력 텐서로 변환한다."""
        if self.tokenizer and "text" in raw_input:
            encoded = self.tokenizer(
                raw_input["text"], return_tensors="pt", padding=True, truncation=True,
            )
            return dict(encoded)

        if self.transform and "image" in raw_input:
            image = raw_input["image"]
            pixel_values = self.transform(image).unsqueeze(0)
            return {"pixel_values": pixel_values}

        return raw_input

    async def start(self) -> None:
        """배치 루프를 시작한다. FastAPI lifespan에서 호출."""
        if self._use_batching:
            self._task = asyncio.create_task(self.scheduler.batch_loop())

    async def stop(self) -> None:
        if hasattr(self, "_task"):
            self._task.cancel()


class _BatchScheduler:
    """짧은 시간 창 내 도착한 요청을 모아 배치 처리한다."""

    def __init__(
        self,
        model: torch.nn.Module,
        forward_fn: Callable[[dict], dict] | None = None,
        max_batch_size: int = 8,
        batch_window_ms: float = 50,
    ) -> None:
        self.model = model
        self.forward_fn = forward_fn or make_forward_fn(model)
        self.max_batch_size = max_batch_size
        self.batch_window = batch_window_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self._device = next(model.parameters()).device

    async def submit(self, preprocessed: dict) -> dict:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((preprocessed, future))
        return await future

    async def batch_loop(self) -> None:
        while True:
            try:
                first_item = await self.queue.get()
                batch_items = [first_item]

                deadline = asyncio.get_running_loop().time() + self.batch_window
                while len(batch_items) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch_items.append(item)
                    except asyncio.TimeoutError:
                        break

                inputs = [item[0] for item in batch_items]
                batched = self._collate(inputs)
                batched = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batched.items()}

                with torch.no_grad():
                    outputs = self.forward_fn(batched)

                if isinstance(outputs, torch.Tensor):
                    outputs = {"logits": outputs}

                for i, (_, future) in enumerate(batch_items):
                    result = self._unbatch(outputs, i)
                    if not future.done():
                        future.set_result(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("배치 처리 실패: %s", e)
                for _, future in batch_items:
                    if not future.done():
                        future.set_exception(e)

    @staticmethod
    def _collate(inputs: list[dict]) -> dict:
        """개별 입력을 배치 텐서로 합친다."""
        keys = inputs[0].keys()
        batched = {}
        for k in keys:
            values = [inp[k] for inp in inputs]
            if isinstance(values[0], torch.Tensor):
                if values[0].dim() >= 1 and any(v.shape != values[0].shape for v in values):
                    # Re-pad to max length (variable-length text support)
                    max_len = max(v.shape[-1] for v in values)
                    padded = []
                    for v in values:
                        if v.shape[-1] < max_len:
                            pad_size = max_len - v.shape[-1]
                            padded.append(torch.nn.functional.pad(v, (0, pad_size)))
                        else:
                            padded.append(v)
                    batched[k] = torch.cat(padded, dim=0)
                else:
                    batched[k] = torch.cat(values, dim=0)
            else:
                batched[k] = values
        return batched

    @staticmethod
    def _unbatch(outputs: dict, index: int) -> dict:
        """배치 출력에서 단일 샘플을 추출한다."""
        result = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                val = v[index]
                if k == "logits":
                    probs = torch.softmax(val, dim=-1)
                    result["prediction"] = int(torch.argmax(val).item())
                    result["probabilities"] = probs.cpu().tolist()
                else:
                    result[k] = val.cpu().tolist()
            else:
                result[k] = v
        return result
