"""GPU 4장 Phase 2 디버깅에서 발견·수정된 RLTrainer 버그 회귀 테스트.

모두 CPU에서 실행 가능한 structural 테스트이다. 실제 학습 루프(multi-GPU FSDP)는
서버에서만 검증 가능하므로, AST 파싱으로 코드 구조를 확인하는 방식을 사용한다.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_RL_TRAINER = Path(__file__).parents[2] / "mdp" / "training" / "rl_trainer.py"
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
