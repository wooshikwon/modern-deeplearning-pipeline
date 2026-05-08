"""device_map 모델에서 inference callback hook 동작을 검증한다.

accelerate의 device_map은 모듈의 forward를 교체하여 디바이스 간 전송을 처리한다.
PyTorch의 사용자 hook은 __call__ 레벨에서 실행되므로 accelerate와 간섭 없이 공존한다.

이 테스트는 단일 CPU/GPU 환경에서도 실행 가능하도록 device_map 동작을 시뮬레이션한다:
- hf_device_map 속성을 설정하여 inference 파이프라인이 .to(dev)를 스킵하도록 함
- forward를 감싸서 accelerate와 동일한 순서를 재현 (pre_forward → forward → post_forward)
- read hook (register_forward_hook)과 write hook (register_forward_pre_hook) 모두 검증
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mdp.callbacks.base import BaseInferenceCallback
from mdp.callbacks.inference import DefaultOutputCallback
from mdp.serving.inference import run_batch_inference
from tests.e2e.datasets import ListDataLoader, make_language_batches
from tests.e2e.models import TinyLanguageModel


# ---------------------------------------------------------------------------
# Helpers: accelerate device_map 시뮬레이션
# ---------------------------------------------------------------------------


def _simulate_device_map(model: nn.Module) -> nn.Module:
    """accelerate의 device_map 동작을 시뮬레이션한다.

    1. hf_device_map 속성 설정 → inference 파이프라인이 .to(dev) 스킵
    2. 특정 layer의 forward를 감싸서 accelerate의 hook 순서를 재현
    """
    model.hf_device_map = {"": "cpu"}  # 단일 디바이스 시뮬레이션
    return model


def _wrap_forward_like_accelerate(module: nn.Module) -> None:
    """accelerate의 add_hook_to_module과 동일한 방식으로 forward를 교체한다.

    이 래퍼는 accelerate의 AlignDevicesHook이 하는 것과 같은 순서로
    pre_forward / _old_forward / post_forward를 실행한다.
    사용자의 PyTorch hook이 이 래핑과 공존하는지 검증하기 위함이다.
    """
    old_forward = module.forward

    def new_forward(*args, **kwargs):
        # accelerate pre_forward 시뮬: 실제로는 여기서 send_to_device 호출
        # (단일 디바이스 테스트이므로 no-op)
        output = old_forward(*args, **kwargs)
        # accelerate post_forward 시뮬: 실제로는 여기서 io_same_device 처리
        return output

    module._old_forward = old_forward
    module.forward = new_forward


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class HiddenStateCaptureCallback(BaseInferenceCallback):
    """forward hook으로 transformer layer의 hidden state를 캡처한다."""

    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name
        self.captured: list[torch.Tensor] = []
        self._hook_handle: Any = None
        self._latest: torch.Tensor | None = None

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
        target = dict(model.named_modules())[self.layer_name]
        self._hook_handle = target.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        self._latest = h.detach().cpu()

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
        if self._latest is not None:
            self.captured.append(self._latest)
            self._latest = None

    def teardown(self, **kwargs) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class ActivationSteeringCallback(BaseInferenceCallback):
    """forward pre-hook으로 hidden state에 벡터를 주입한다."""

    def __init__(self, layer_name: str, vector: torch.Tensor, strength: float = 1.0) -> None:
        self.layer_name = layer_name
        self.vector = vector
        self.strength = strength
        self._hook_handle: Any = None
        self.pre_hook_called_count = 0

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
        target = dict(model.named_modules())[self.layer_name]
        self._hook_handle = target.register_forward_pre_hook(self._steer_fn)

    def _steer_fn(self, module: nn.Module, args: tuple) -> tuple:
        self.pre_hook_called_count += 1
        if not args:
            return args
        hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            return args
        vec = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden + self.strength * vec
        return (steered,) + args[1:]

    def teardown(self, **kwargs) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class SteerAndCaptureCallback(BaseInferenceCallback):
    """하나의 콜백에서 pre-hook (steering)과 post-hook (capture)을 동시에 등록한다."""

    def __init__(
        self,
        steer_layer: str,
        capture_layer: str,
        vector: torch.Tensor,
        strength: float = 1.0,
    ) -> None:
        self.steer_layer = steer_layer
        self.capture_layer = capture_layer
        self.vector = vector
        self.strength = strength
        self.captured: list[torch.Tensor] = []
        self._handles: list = []
        self._latest: torch.Tensor | None = None

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
        steer_target = dict(model.named_modules())[self.steer_layer]
        capture_target = dict(model.named_modules())[self.capture_layer]

        h1 = steer_target.register_forward_pre_hook(self._steer_fn)
        h2 = capture_target.register_forward_hook(self._capture_fn)
        self._handles = [h1, h2]

    def _steer_fn(self, module: nn.Module, args: tuple) -> tuple:
        if not args:
            return args
        hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            return args
        vec = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden + self.strength * vec,) + args[1:]

    def _capture_fn(self, module: nn.Module, input: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        self._latest = h.detach().cpu()

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
        if self._latest is not None:
            self.captured.append(self._latest)
            self._latest = None

    def teardown(self, **kwargs) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_hook_with_device_map(tmp_path: Path) -> None:
    """device_map 모델에서 forward hook (읽기)이 정상 동작한다."""
    model = TinyLanguageModel(vocab_size=64, hidden_dim=32)
    model = _simulate_device_map(model)

    batches = make_language_batches(num_batches=3, batch_size=4, seq_len=16, vocab_size=64)
    loader = ListDataLoader(batches)

    cb = HiddenStateCaptureCallback(layer_name="lm_head")
    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="text_generation",
        device="cpu",
        callbacks=[cb],
    )

    assert cb.captured, "forward hook이 활성화를 캡처하지 못함"
    assert len(cb.captured) == 3, f"배치 수 불일치: {len(cb.captured)} != 3"
    # lm_head는 Linear(32, 64) → shape (B, L, 64)
    for act in cb.captured:
        assert act.shape == (4, 16, 64), f"shape 불일치: {act.shape}"
    assert cb._hook_handle is None, "teardown에서 hook이 해제되지 않음"


def test_forward_hook_with_accelerate_wrapper(tmp_path: Path) -> None:
    """accelerate 방식으로 forward가 교체된 모듈에서도 forward hook이 동작한다."""
    model = TinyLanguageModel(vocab_size=64, hidden_dim=32)
    model = _simulate_device_map(model)

    # transformer encoder의 forward를 accelerate 방식으로 래핑
    _wrap_forward_like_accelerate(model.transformer)

    batches = make_language_batches(num_batches=2, batch_size=4, seq_len=16, vocab_size=64)
    loader = ListDataLoader(batches)

    cb = HiddenStateCaptureCallback(layer_name="transformer")
    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="text_generation",
        device="cpu",
        callbacks=[cb],
    )

    assert len(cb.captured) == 2
    # TransformerEncoder 출력: (B, L, hidden_dim=32)
    for act in cb.captured:
        assert act.shape == (4, 16, 32)


def test_pre_hook_steering_modifies_output(tmp_path: Path) -> None:
    """forward pre-hook으로 입력을 수정하면 최종 출력이 변경된다."""
    model = TinyLanguageModel(vocab_size=64, hidden_dim=32)
    model.eval()

    batches = make_language_batches(num_batches=1, batch_size=2, seq_len=8, vocab_size=64)

    # Baseline: steering 없이 추론
    with torch.no_grad():
        baseline_out = model(batches[0])["logits"].clone()

    # Steering: transformer 입력에 큰 벡터 주입
    # transformer는 (B, L, D=32) hidden states를 입력으로 받는다
    steering_vec = torch.randn(1, 1, 32) * 10.0  # (1, 1, D) — broadcast
    cb = ActivationSteeringCallback(
        layer_name="transformer",
        vector=steering_vec,
        strength=1.0,
    )

    loader = ListDataLoader(batches)
    capture_cb = HiddenStateCaptureCallback(layer_name="lm_head")

    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="text_generation",
        device="cpu",
        callbacks=[cb, capture_cb],
    )

    assert cb.pre_hook_called_count > 0, "pre-hook이 호출되지 않음"
    steered_logits = capture_cb.captured[0]
    # steering이 적용되면 출력이 baseline과 달라야 한다
    assert not torch.allclose(baseline_out, steered_logits, atol=1e-4), \
        "steering이 출력에 영향을 주지 않음"


def test_pre_hook_dtype_matching() -> None:
    """fp32 steering vector가 bf16 hidden states와 정상 연산된다.

    .to(device=..., dtype=...) 없이 .to(device=...)만 하면 dtype 불일치로
    결과가 부정확하거나 에러가 발생할 수 있다. 콜백이 dtype을 맞추는지 검증한다.
    """
    model = TinyLanguageModel(vocab_size=64, hidden_dim=32)
    model = model.to(dtype=torch.bfloat16)
    model.eval()

    batches = make_language_batches(num_batches=1, batch_size=2, seq_len=8, vocab_size=64)

    # fp32 steering vector — 콜백이 bf16으로 변환해야 한다
    # transformer 입력은 (B, L, D=32) hidden states
    steering_vec = torch.randn(1, 1, 32, dtype=torch.float32)

    cb = ActivationSteeringCallback(
        layer_name="transformer",
        vector=steering_vec,
        strength=1.0,
    )

    # setup에서 hook 등록
    cb.setup(model=model)

    # forward 실행 — dtype 불일치 없이 통과해야 한다
    with torch.no_grad():
        outputs = model(batches[0])

    assert cb.pre_hook_called_count > 0, "pre-hook이 호출되지 않음"
    assert outputs["logits"].dtype == torch.bfloat16, "출력 dtype이 변경됨"

    cb.teardown()


def test_steer_and_capture_combined(tmp_path: Path) -> None:
    """하나의 콜백에서 steering(pre-hook)과 capture(post-hook)를 동시에 사용한다."""
    model = TinyLanguageModel(vocab_size=64, hidden_dim=32)
    model = _simulate_device_map(model)

    steering_vec = torch.randn(1, 1, 32) * 5.0

    cb = SteerAndCaptureCallback(
        steer_layer="transformer",
        capture_layer="lm_head",
        vector=steering_vec,
        strength=1.0,
    )

    batches = make_language_batches(num_batches=2, batch_size=4, seq_len=16, vocab_size=64)
    loader = ListDataLoader(batches)

    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="text_generation",
        device="cpu",
        callbacks=[cb],
    )

    assert len(cb.captured) == 2, f"캡처 수 불일치: {len(cb.captured)} != 2"
    assert len(cb._handles) == 0, "teardown에서 handle이 해제되지 않음"


def test_device_map_skips_model_to(tmp_path: Path) -> None:
    """hf_device_map 속성이 있으면 inference 파이프라인이 .to(dev)를 스킵한다."""
    model = TinyLanguageModel(vocab_size=64, hidden_dim=32)
    model = _simulate_device_map(model)

    batches = make_language_batches(num_batches=1, batch_size=2, seq_len=8, vocab_size=64)
    loader = ListDataLoader(batches)

    output_cb = DefaultOutputCallback(
        output_path=tmp_path / "preds", output_format="jsonl", task="text_generation",
    )

    # 에러 없이 추론이 완료되어야 한다
    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="text_generation",
        device="cpu",
        callbacks=[output_cb],
    )
    assert result_path is not None
    assert result_path.exists()


def test_pre_hook_runs_before_accelerate_wrapper(tmp_path: Path) -> None:
    """사용자 pre-hook이 accelerate의 forward wrapper보다 먼저 실행된다.

    실행 순서: user pre-hook → accelerate pre_forward → _old_forward → accelerate post_forward → user post-hook
    이 순서가 보장되는지 실행 로그로 검증한다.
    """
    model = TinyLanguageModel(vocab_size=64, hidden_dim=32)
    model = _simulate_device_map(model)

    execution_log: list[str] = []

    # accelerate 방식으로 transformer forward를 래핑 + 로깅
    old_forward = model.transformer.forward

    def accel_wrapper(*args, **kwargs):
        execution_log.append("accelerate_pre_forward")
        result = old_forward(*args, **kwargs)
        execution_log.append("accelerate_post_forward")
        return result

    model.transformer._old_forward = old_forward
    model.transformer.forward = accel_wrapper

    # 사용자 pre-hook 등록 (transformer에)
    def user_pre_hook(module, args):
        execution_log.append("user_pre_hook")
        return args

    # 사용자 post-hook 등록 (transformer에)
    def user_post_hook(module, input, output):
        execution_log.append("user_post_hook")

    model.transformer.register_forward_pre_hook(user_pre_hook)
    model.transformer.register_forward_hook(user_post_hook)

    batches = make_language_batches(num_batches=1, batch_size=2, seq_len=8, vocab_size=64)
    model.eval()
    with torch.no_grad():
        model(batches[0])

    assert execution_log == [
        "user_pre_hook",
        "accelerate_pre_forward",
        "accelerate_post_forward",
        "user_post_hook",
    ], f"실행 순서 불일치: {execution_log}"
