"""generate 경로에서 BaseInferenceCallback lifecycle dispatch 검증."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from mdp.callbacks.base import BaseInferenceCallback
from mdp.cli.output import ModelSourcePlan


_PRETRAINED_SOURCE_PLAN = ModelSourcePlan(
    kind="pretrained",
    command="generate",
    uri="hf://fake-model",
    supports_pretrained=True,
)


class _TrackingCallback(BaseInferenceCallback):
    """setup/on_batch/teardown 호출을 추적하는 테스트용 콜백."""

    def __init__(self) -> None:
        self.setup_called = False
        self.on_batch_calls: list[int] = []
        self.teardown_called = False

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
        self.setup_called = True

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
        self.on_batch_calls.append(batch_idx)
        assert "generated_ids" in outputs

    def teardown(self, **kwargs) -> None:
        self.teardown_called = True


class _FakeGenerativeModel(nn.Module):
    """model.generate()를 지원하는 가짜 모델."""

    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)  # parameters()가 비어있지 않도록
        self.vocab_size = vocab_size

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        max_new = kwargs.get("max_new_tokens", 4)
        new_ids = torch.randint(0, self.vocab_size, (batch_size, max_new))
        return torch.cat([input_ids, new_ids], dim=1)


class _FakeTokenizer:
    """HuggingFace AutoTokenizer를 대체하는 가짜 토크나이저."""

    pad_token: str | None = None
    eos_token: str = "<eos>"
    pad_token_id: int = 0

    def __call__(self, texts: list[str], **kwargs) -> dict:
        input_ids = torch.ones(len(texts), 4, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        result = MagicMock()
        result.__getitem__ = lambda self, key: {"input_ids": input_ids, "attention_mask": attention_mask}[key]
        result.__contains__ = lambda self, key: key in {"input_ids", "attention_mask"}
        result.keys = lambda: ["input_ids", "attention_mask"]
        result.to = lambda device: {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
        return result

    def batch_decode(self, ids: torch.Tensor, **kwargs) -> list[str]:
        return [f"generated_{i}" for i in range(ids.shape[0])]

    @classmethod
    def from_pretrained(cls, name: str) -> "_FakeTokenizer":
        return cls()


def _make_prompts_file(tmp_path: Path, n: int = 3) -> Path:
    """n개 프롬프트가 담긴 JSONL 파일을 생성한다."""
    prompts_path = tmp_path / "prompts.jsonl"
    with open(prompts_path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"prompt": f"Hello {i}"}) + "\n")
    return prompts_path


def test_generate_callback_lifecycle(tmp_path: Path) -> None:
    """run_generate에서 BaseInferenceCallback의 setup/on_batch/teardown이 dispatch된다."""
    from mdp.cli.generate import run_generate

    model = _FakeGenerativeModel()
    tokenizer = _FakeTokenizer()
    cb = _TrackingCallback()
    prompts_file = _make_prompts_file(tmp_path, n=3)

    with (
        patch("mdp.cli.generate.resolve_model_source_plan", return_value=_PRETRAINED_SOURCE_PLAN),
        patch("mdp.models.pretrained.PretrainedResolver.load", return_value=model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("mdp.training._common.load_callbacks_from_file", return_value=[{"_component_": "Dummy"}]),
        patch("mdp.training._common.create_callbacks", return_value=[cb]),
    ):
        run_generate(
            run_id=None,
            model_dir=None,
            prompts=str(prompts_file),
            output=str(tmp_path / "out.jsonl"),
            pretrained="hf://fake-model",
            callbacks_file="fake.yaml",
            max_new_tokens=4,
            batch_size=2,
        )

    assert cb.setup_called, "setup must be called before generation"
    assert cb.teardown_called, "teardown must be called after generation"
    # batch_size=2, 3 prompts → 2 batches (2+1), num_samples=1 → 2 on_batch calls
    assert len(cb.on_batch_calls) == 2
    assert cb.on_batch_calls == [0, 1]


def test_generate_callback_teardown_on_error(tmp_path: Path) -> None:
    """생성 루프 에러 시에도 teardown은 실행된다."""
    from mdp.cli.generate import run_generate

    class _ExplodingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self._call_count = 0

        def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
            self._call_count += 1
            if self._call_count >= 2:
                raise RuntimeError("deliberate explosion")
            batch_size, seq_len = input_ids.shape
            return torch.cat([input_ids, torch.ones(batch_size, 2, dtype=torch.long)], dim=1)

    model = _ExplodingModel()
    tokenizer = _FakeTokenizer()
    cb = _TrackingCallback()
    prompts_file = _make_prompts_file(tmp_path, n=4)

    import typer

    with (
        patch("mdp.cli.generate.resolve_model_source_plan", return_value=_PRETRAINED_SOURCE_PLAN),
        patch("mdp.models.pretrained.PretrainedResolver.load", return_value=model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("mdp.training._common.load_callbacks_from_file", return_value=[{"_component_": "Dummy"}]),
        patch("mdp.training._common.create_callbacks", return_value=[cb]),
    ):
        try:
            run_generate(
                run_id=None,
                model_dir=None,
                prompts=str(prompts_file),
                output=str(tmp_path / "out.jsonl"),
                pretrained="hf://fake-model",
                callbacks_file="fake.yaml",
                max_new_tokens=2,
                batch_size=2,
            )
        except (RuntimeError, typer.Exit):
            pass

    assert cb.teardown_called, "teardown must run even when generation loop raises"


def test_generate_callback_teardown_on_pre_loop_error(tmp_path: Path) -> None:
    """setup 후 생성 루프 진입 전 에러(빈 프롬프트 등)에서도 teardown이 실행된다."""
    from mdp.cli.generate import run_generate

    model = _FakeGenerativeModel()
    tokenizer = _FakeTokenizer()
    cb = _TrackingCallback()

    # 빈 프롬프트 파일 → ValueError("프롬프트 파일이 비어 있습니다")
    empty_prompts = tmp_path / "empty.jsonl"
    empty_prompts.write_text("")

    import typer

    with (
        patch("mdp.cli.generate.resolve_model_source_plan", return_value=_PRETRAINED_SOURCE_PLAN),
        patch("mdp.models.pretrained.PretrainedResolver.load", return_value=model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("mdp.training._common.load_callbacks_from_file", return_value=[{"_component_": "Dummy"}]),
        patch("mdp.training._common.create_callbacks", return_value=[cb]),
    ):
        try:
            run_generate(
                run_id=None,
                model_dir=None,
                prompts=str(empty_prompts),
                output=str(tmp_path / "out.jsonl"),
                pretrained="hf://fake-model",
                callbacks_file="fake.yaml",
                max_new_tokens=4,
            )
        except (ValueError, typer.Exit):
            pass

    assert cb.setup_called, "setup must be called before the error"
    assert cb.teardown_called, "teardown must run even when pre-loop validation raises"
    assert len(cb.on_batch_calls) == 0, "on_batch must not be called if loop never started"


def test_generate_no_callbacks(tmp_path: Path) -> None:
    """callbacks_file=None일 때 콜백 없이 정상 동작한다."""
    from mdp.cli.generate import run_generate

    model = _FakeGenerativeModel()
    tokenizer = _FakeTokenizer()
    prompts_file = _make_prompts_file(tmp_path, n=2)

    with (
        patch("mdp.cli.generate.resolve_model_source_plan", return_value=_PRETRAINED_SOURCE_PLAN),
        patch("mdp.models.pretrained.PretrainedResolver.load", return_value=model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        run_generate(
            run_id=None,
            model_dir=None,
            prompts=str(prompts_file),
            output=str(tmp_path / "out.jsonl"),
            pretrained="hf://fake-model",
            callbacks_file=None,
            max_new_tokens=4,
        )

    output_path = tmp_path / "out.jsonl"
    assert output_path.exists()
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2
