"""서빙 핸들러 — task별 요청 처리 전략."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable

import torch
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


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

    async def handle(self, raw_input: dict) -> StreamingResponse:
        text = raw_input.get("text", "")
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        streamer = self._streamer_cls(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": self.generation_config.get("max_new_tokens", 256),
            "temperature": self.generation_config.get("temperature", 1.0),
            "do_sample": self.generation_config.get("do_sample", True),
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        async def event_stream():
            for token_text in streamer:
                yield f"data: {json.dumps({'token': token_text})}\n\n"
            yield "data: [DONE]\n\n"

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
        max_bs = 1
        window = 50.0
        if serving_config is not None:
            max_bs = getattr(serving_config, "max_batch_size", 1)
            window = getattr(serving_config, "batch_window_ms", 50.0)
        self._use_batching = max_bs > 1
        if self._use_batching:
            self.scheduler = _BatchScheduler(model, max_batch_size=max_bs, batch_window_ms=window)

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
            return self.model(inputs)

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
        max_batch_size: int = 8,
        batch_window_ms: float = 50,
    ) -> None:
        self.model = model
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
                    outputs = self.model(batched)

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
