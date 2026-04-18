"""`mdp/_liger_patch.py` 헬퍼의 단위 테스트.

Liger-Kernel은 CUDA 전용 optional dependency라 CPU CI 환경에서는 보통 미설치.
본 테스트는 import 실패 시 silent skip(False 반환), 이미 적용된 상태 재호출 시
idempotent(False 반환)를 검증한다. Liger가 실제 설치되어 있을 때의 성공 경로는
monkeypatch로 가짜 모듈을 sys.modules에 주입해 시뮬레이션한다.

spec-algorithm-hidden-states-support §U2 verify 기준:
- CI(CPU) 환경 전체 pytest 회귀 0 (liger-kernel 미설치 시 ImportError로 silent pass).
- import path 정확성, try/except 구조, log statement 존재 유무 검증.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _reset_applied_flag():
    """각 테스트 전후로 _APPLIED module-level 가드를 초기화한다.

    apply_liger_patches는 한 번 성공하면 재호출 시 False를 반환하는 idempotent
    설계라 테스트 간 오염을 막기 위해 매번 reset한다.
    """
    import mdp._liger_patch as lp
    lp._APPLIED = False
    yield
    lp._APPLIED = False


def test_apply_liger_patches_missing_returns_false(monkeypatch, caplog):
    """Liger 미설치 환경에서는 False를 반환하고 debug 로그만 남긴다.

    CPU CI의 정상 경로이므로 WARNING/ERROR 로그가 발생하면 안 된다.
    """
    # liger_kernel 모듈이 import되지 않도록 sys.modules에서 강제 제거하고
    # 재import 시 ImportError를 유발하도록 finder를 조작한다.
    for name in list(sys.modules):
        if name.startswith("liger_kernel"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    # liger_kernel을 명시적으로 None으로 표시해 import가 실패하도록 한다.
    monkeypatch.setitem(sys.modules, "liger_kernel", None)

    from mdp._liger_patch import apply_liger_patches

    with caplog.at_level(logging.DEBUG, logger="mdp._liger_patch"):
        result = apply_liger_patches()

    assert result is False
    # WARNING 이상 로그가 없어야 한다 (silent skip).
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f"Expected no warnings on missing liger, got: {warnings}"


def test_apply_liger_patches_success_returns_true(monkeypatch, caplog):
    """가짜 liger_kernel 모듈을 주입했을 때 patch가 적용되고 INFO 로그가 남는다.

    review-2026-04-18-U6-c1 §2-1 회귀 고정: Liger 기본 flag 다섯 가지
    (rope / cross_entropy / fused_linear_cross_entropy / rms_norm / swiglu)를
    **명시적으로** 지정한다. spec §U2 "FLCE only" 원칙을 위반하지 않도록
    기본값(True)을 False로 덮어쓰는 것이 핵심.
    """
    calls: list[dict] = []

    fake_transformers = types.ModuleType("liger_kernel.transformers")

    def fake_apply(**kwargs):
        calls.append(kwargs)

    fake_transformers.apply_liger_kernel_to_llama = fake_apply

    fake_root = types.ModuleType("liger_kernel")
    fake_root.transformers = fake_transformers

    monkeypatch.setitem(sys.modules, "liger_kernel", fake_root)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", fake_transformers)

    from mdp._liger_patch import apply_liger_patches

    with caplog.at_level(logging.INFO, logger="mdp._liger_patch"):
        result = apply_liger_patches()

    assert result is True
    # 2-1 핵심: 5개 flag를 전부 명시 호출.
    assert len(calls) == 1, f"apply_liger_kernel_to_llama must be called exactly once, got {calls}"
    kwargs = calls[0]
    assert kwargs == {
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": False,
        "swiglu": False,
    }, f"spec §U2 'FLCE only' flag set mismatch: {kwargs}"
    assert any(
        "Liger kernel applied" in r.message and r.levelno == logging.INFO
        for r in caplog.records
    ), "INFO log for successful patch not found"


def test_apply_liger_patches_is_idempotent(monkeypatch):
    """이미 적용된 후 재호출하면 False를 반환하고 fake_apply를 재호출하지 않는다."""
    calls: list[dict] = []

    fake_transformers = types.ModuleType("liger_kernel.transformers")

    def fake_apply(**kwargs):
        calls.append(kwargs)

    fake_transformers.apply_liger_kernel_to_llama = fake_apply

    fake_root = types.ModuleType("liger_kernel")
    fake_root.transformers = fake_transformers

    monkeypatch.setitem(sys.modules, "liger_kernel", fake_root)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", fake_transformers)

    from mdp._liger_patch import apply_liger_patches

    first = apply_liger_patches()
    second = apply_liger_patches()
    third = apply_liger_patches()

    assert first is True
    assert second is False
    assert third is False
    # idempotent: 실제 patch 호출은 한 번만.
    assert len(calls) == 1


def test_apply_liger_patches_handles_exception(monkeypatch, caplog):
    """Liger가 설치되어 있으나 apply 호출에서 예외가 나면 WARNING 로그 후 False."""
    fake_transformers = types.ModuleType("liger_kernel.transformers")

    def fake_apply(**kwargs):
        raise RuntimeError("incompatible transformers version")

    fake_transformers.apply_liger_kernel_to_llama = fake_apply

    fake_root = types.ModuleType("liger_kernel")
    fake_root.transformers = fake_transformers

    monkeypatch.setitem(sys.modules, "liger_kernel", fake_root)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", fake_transformers)

    from mdp._liger_patch import apply_liger_patches

    with caplog.at_level(logging.WARNING, logger="mdp._liger_patch"):
        result = apply_liger_patches()

    assert result is False
    assert any(
        "Liger monkey-patch failed" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_cli_entries_call_apply_liger_patches():
    """3개 CLI entry에서 apply_liger_patches가 import되어 호출되는지 source 수준 검증.

    실제 학습 파이프라인을 돌리지 않고, 모듈의 소스 텍스트에서 호출 위치를 확인한다.
    spec §U2의 "import path 정확성 + log statement 존재 유무" 검증 요건 충족.
    """
    from pathlib import Path

    import mdp

    mdp_root = Path(mdp.__file__).parent
    targets = {
        "_torchrun_entry": mdp_root / "cli" / "_torchrun_entry.py",
        "train": mdp_root / "cli" / "train.py",
        "rl_train": mdp_root / "cli" / "rl_train.py",
    }

    for name, path in targets.items():
        text = path.read_text()
        assert "apply_liger_patches" in text, (
            f"{name} ({path}) does not reference apply_liger_patches"
        )
        assert "from mdp._liger_patch import" in text, (
            f"{name} ({path}) does not import from mdp._liger_patch"
        )
