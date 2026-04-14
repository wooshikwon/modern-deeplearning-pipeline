"""GPU 4장 Phase 2 디버깅에서 발견·수정된 RLTrainer / Trainer 버그 회귀 테스트.

모두 CPU에서 실행 가능한 structural 테스트이다. 실제 학습 루프(multi-GPU FSDP)는
서버에서만 검증 가능하므로, AST 파싱으로 코드 구조를 확인하는 방식을 사용한다.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_RL_TRAINER = Path(__file__).parents[2] / "mdp" / "training" / "rl_trainer.py"
_TRAINER = Path(__file__).parents[2] / "mdp" / "training" / "trainer.py"
_PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


# ── Bug 1: peft 0.19.0 + torch 2.5.1 버전 경계 충돌 ──
#
# peft 0.19.0의 tuners_utils.py::cast_adapter_dtype이 UPCAST_DTYPES 목록을 순회하며
# getattr(torch, name)을 호출하는데, torch.float8_e8m0fnu는 torch 2.6에서 추가된
# dtype이라 torch 2.5.1에 없다 → AttributeError.


def test_peft_upper_bound_in_pyproject() -> None:
    """pyproject.toml [language] 의존성에 peft<0.19.0 상한이 있어야 한다."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with _PYPROJECT.open("rb") as f:
        data = tomllib.load(f)

    lang_deps = data["project"]["optional-dependencies"]["language"]
    peft_entry = next((d for d in lang_deps if d.startswith("peft")), None)
    assert peft_entry is not None, "peft가 [language] 의존성 목록에 없다"
    assert "<0.19.0" in peft_entry, (
        f"peft 의존성에 <0.19.0 상한이 없다: {peft_entry!r}. "
        "peft 0.19.0은 torch.float8_e8m0fnu를 참조하는데 torch 2.5.1에 없어 "
        "AttributeError가 발생한다."
    )


# ── Bug 2: UnboundLocalError — train() 내부의 `import torch` ──
#
# Python scoping rule: 함수 내 어느 위치든 변수에 assign(import 포함)이 있으면
# 그 함수 전체에서 그 이름이 local variable로 취급된다.
# torch.compile 블록 안에 `import torch`가 있으면, 그 위에서 torch를 참조하는
# torch.cuda.is_available() 등이 UnboundLocalError를 낸다.


def test_no_local_import_torch_in_train_method() -> None:
    """rl_trainer.py::RLTrainer.train() 내에 `import torch`가 없어야 한다."""
    source = _RL_TRAINER.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RLTrainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "train":
                    for child in ast.walk(method):
                        if isinstance(child, ast.Import):
                            for alias in child.names:
                                assert alias.name != "torch", (
                                    f"train() 내부 line {child.lineno}에 `import torch`가 있다. "
                                    "Python은 함수 내 assign이 있으면 전체를 local로 취급하므로, "
                                    "이 import보다 위에서 torch를 참조하면 UnboundLocalError가 발생한다. "
                                    "module-level import(line 18)만 사용해야 한다."
                                )


# ── Bug 3: AttributeError — compile_mode가 __init__에 없음 ──
#
# train() 메서드에 `if self.compile_mode:` 분기가 있는데, __init__에서
# self.compile_mode를 초기화하지 않으면 AttributeError가 발생한다.
# trainer.py는 __init__에서 올바르게 초기화하지만 rl_trainer.py는 누락됐었다.


def test_rl_trainer_init_assigns_compile_mode() -> None:
    """RLTrainer.__init__이 self.compile_mode를 초기화해야 한다."""
    source = _RL_TRAINER.read_text()
    tree = ast.parse(source)

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RLTrainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                    for child in ast.walk(method):
                        if (
                            isinstance(child, ast.Assign)
                            and any(
                                isinstance(t, ast.Attribute) and t.attr == "compile_mode"
                                for t in child.targets
                            )
                        ):
                            found = True
    assert found, (
        "RLTrainer.__init__에 `self.compile_mode = ...` 할당이 없다. "
        "train()의 `if self.compile_mode:` 블록이 AttributeError를 낸다."
    )


# ── Bug 4: OOM — model.train()이 FSDP setup 후 호출되지 않음 ──
#
# HF from_pretrained()는 eval() 모드로 모델을 반환한다. FSDP wrap은 training
# mode를 변경하지 않는다. LlamaModel.forward의 GC guard는
# `self.gradient_checkpointing and self.training`이므로 self.training=False이면
# GC가 비활성화되어 32 레이어 전체 activation이 저장된다 → OOM.
# trainer.py는 _train_one_epoch()가 매 에폭마다 model.train()을 호출하지만,
# rl_trainer.py는 독립 루프라 이 호출이 없었다.


def test_train_method_sets_training_mode_after_strategy_setup() -> None:
    """train()이 strategy setup 후 trainable 모델에 .train()을, frozen 모델에 .eval()을 호출한다."""
    source = _RL_TRAINER.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RLTrainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "train":
                    method_src = ast.get_source_segment(source, method) or ""

                    assert "self.trainable.values()" in method_src and ".train()" in method_src, (
                        "train()에서 trainable 모델들에 .train()을 호출하는 코드가 없다. "
                        "HF 모델은 from_pretrained() 후 eval() 상태이므로 명시적 설정이 필요하다."
                    )
                    assert "self.frozen.values()" in method_src and ".eval()" in method_src, (
                        "train()에서 frozen 모델들에 .eval()을 호출하는 코드가 없다."
                    )


# ── Validation mode restore — policy.train() 복원 ──
#
# _run_rl_validation / _run_dpo_validation은 self.policy.eval()로 전환한 후
# 마지막에 self.policy.train()으로 복원해야 한다. 복원이 없으면 validation 이후
# 학습 루프가 eval 모드로 계속 실행된다.


@pytest.mark.parametrize("method_name", ["_run_rl_validation", "_run_dpo_validation"])
def test_validation_methods_restore_policy_train_mode(method_name: str) -> None:
    """validation 메서드들이 완료 후 self.policy.train()을 호출해야 한다."""
    source = _RL_TRAINER.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RLTrainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    method_src = ast.get_source_segment(source, method) or ""
                    assert "self.policy.train()" in method_src, (
                        f"{method_name}()가 self.policy.train()으로 training mode를 복원하지 않는다. "
                        "validation 이후 학습 루프가 eval 모드로 실행된다."
                    )


# ── Trainer GC 경로 회귀 (trainer.py와 rl_trainer.py의 대칭성) ──
#
# RLTrainer는 올바르게 구현되어 있었으나 Trainer는 다음 3가지가 빠져 있었다:
# (A) GC를 FSDP wrap 이후에 적용 → use_reentrant=True(기본)가 FSDP param 조기 해제를 막아 OOM 가능
# (B) use_reentrant=False 미지정 → FSDP + GC 호환성 문제
# (C) enable_input_require_grads 미호출 → LoRA + GC에서 gradient가 recompute 구간에서 소실 (silent failure)


def test_trainer_gc_uses_use_reentrant_false() -> None:
    """Trainer.train()의 GC 활성화 코드에 use_reentrant=False가 명시되어야 한다.

    FSDP + GC: use_reentrant=True(기본값)는 FSDP가 backward recompute 중
    all-gathered 파라미터를 조기에 해제하지 못해 전 레이어 파라미터가 동시
    상주 → OOM. use_reentrant=False(비재진입)로 FSDP 호환성을 확보한다.
    """
    source = _TRAINER.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "train":
                    method_src = ast.get_source_segment(source, method) or ""
                    assert "use_reentrant" in method_src, (
                        "Trainer.train()의 GC 코드에 use_reentrant 설정이 없다. "
                        "FSDP + GC 조합에서 OOM이 발생할 수 있다."
                    )
                    assert "False" in method_src, (
                        "Trainer.train()의 GC 코드에 use_reentrant=False가 없다."
                    )


def test_trainer_gc_calls_enable_input_require_grads() -> None:
    """Trainer.train()의 GC 활성화 코드에 enable_input_require_grads 호출이 있어야 한다.

    LoRA 파라미터는 requires_grad=True이지만 입력 텐서는 그렇지 않다.
    GC recompute 시 입력 텐서에 grad_fn이 없으면 LoRA gradient가 소실된다 (silent failure).
    enable_input_require_grads()는 입력에 hook을 걸어 이 문제를 해결한다.
    """
    source = _TRAINER.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "train":
                    method_src = ast.get_source_segment(source, method) or ""
                    assert "enable_input_require_grads" in method_src, (
                        "Trainer.train()에 enable_input_require_grads() 호출이 없다. "
                        "LoRA + GC 조합에서 gradient가 recompute 구간에서 소실된다."
                    )


def test_trainer_gc_before_strategy_setup() -> None:
    """Trainer.train()에서 GC 활성화가 strategy setup(FSDP wrap)보다 먼저 나와야 한다.

    GC는 FSDP wrap 이전에 적용해야 한다. FSDP wrap 이후에 GC를 활성화하면
    FSDP가 checkpoint boundaries를 인식하지 못해 메모리 최적화가 깨진다.
    """
    source = _TRAINER.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "train":
                    method_src = ast.get_source_segment(source, method) or ""
                    gc_pos = method_src.find("gradient_checkpointing_enable")
                    strategy_pos = method_src.find("self.strategy.setup(")
                    assert gc_pos != -1, "train()에 gradient_checkpointing_enable 호출이 없다"
                    assert strategy_pos != -1, "train()에 strategy.setup() 호출이 없다"
                    assert gc_pos < strategy_pos, (
                        f"GC 활성화(pos={gc_pos})가 strategy.setup()(pos={strategy_pos})보다 뒤에 있다. "
                        "GC는 반드시 FSDP wrap 이전에 적용해야 한다."
                    )
