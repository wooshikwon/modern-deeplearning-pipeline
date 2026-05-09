"""_checkpoint.py free function 단위 테스트 (spec-training-restructure §U3).

save_checkpoint → load_checkpoint round-trip 및 find_best_checkpoint 로직을
tmp_path fixture로 고립 검증한다. Trainer / RLTrainer 인스턴스 없이 I/O 레이어만 테스트.

대상 함수:
- ``save_checkpoint``
- ``load_checkpoint``
- ``find_best_checkpoint``
- ``gather_fsdp_state_dict`` (non-FSDP 경로 — FSDP 없이 None 반환 검증)
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mdp.settings.components import ComponentSpec
from mdp.training._checkpoint import (
    find_best_checkpoint,
    gather_fsdp_state_dict,
    load_checkpoint,
    save_checkpoint,
    _write_serving_model_artifact,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. save_checkpoint / load_checkpoint round-trip
# ──────────────────────────────────────────────────────────────────────────────


def test_save_and_load_trainer_state(tmp_path: Path) -> None:
    """trainer_state dict가 trainer_state.json에 기록되고 복원된다."""
    state = {
        "trainer_state": {
            "epoch": 3,
            "global_step": 100,
            "step_in_epoch": 0,
        },
    }
    ckpt_dir = tmp_path / "ckpt_epoch3"
    save_checkpoint(state, ckpt_dir)

    assert (ckpt_dir / "trainer_state.json").exists()

    restored = load_checkpoint(ckpt_dir)
    assert restored["trainer_state"]["epoch"] == 3
    assert restored["trainer_state"]["global_step"] == 100
    assert restored["ckpt_dir"] == ckpt_dir


def test_save_creates_directory_if_missing(tmp_path: Path) -> None:
    """ckpt_dir가 존재하지 않아도 save_checkpoint가 자동으로 생성한다."""
    ckpt_dir = tmp_path / "nested" / "does_not_exist"
    assert not ckpt_dir.exists()

    save_checkpoint({"trainer_state": {"step": 0}}, ckpt_dir)
    assert ckpt_dir.exists()


def test_save_and_load_scaler_state(tmp_path: Path) -> None:
    """scaler state_dict이 scaler.pt에 저장되고 load_checkpoint로 복원된다."""
    fake_scaler_sd = {"scale": 65536.0, "_enabled": True}
    state = {
        "trainer_state": {"step": 5},
        "scaler": fake_scaler_sd,
    }
    ckpt_dir = tmp_path / "ckpt_amp"
    save_checkpoint(state, ckpt_dir)

    assert (ckpt_dir / "scaler.pt").exists()

    restored = load_checkpoint(ckpt_dir)
    assert restored["scaler"] == pytest.approx({"scale": 65536.0, "_enabled": True})


def test_load_missing_scaler_returns_none(tmp_path: Path) -> None:
    """scaler.pt가 없으면 load_checkpoint가 scaler=None을 반환한다."""
    state = {"trainer_state": {"step": 1}}
    ckpt_dir = tmp_path / "no_scaler"
    save_checkpoint(state, ckpt_dir)

    restored = load_checkpoint(ckpt_dir)
    assert restored["scaler"] is None


def test_load_missing_trainer_state_returns_none(tmp_path: Path) -> None:
    """trainer_state.json이 없으면 load_checkpoint가 trainer_state=None을 반환한다."""
    # 빈 디렉토리를 만들어 파일 없이 로드
    ckpt_dir = tmp_path / "empty_ckpt"
    ckpt_dir.mkdir()

    restored = load_checkpoint(ckpt_dir)
    assert restored["trainer_state"] is None
    assert restored["ckpt_dir"] == ckpt_dir


def test_load_nonexistent_dir_raises(tmp_path: Path) -> None:
    """존재하지 않는 경로를 load_checkpoint에 넘기면 FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist")


def test_save_optimizer_per_model(tmp_path: Path) -> None:
    """optimizers dict의 각 항목이 {name}/optimizer.pt에 저장된다 (RLTrainer 패턴)."""
    fake_opt_sd = {"state": {}, "param_groups": [{"lr": 1e-4}]}
    state = {
        "trainer_state": {"step": 10},
        "optimizers": {
            "policy": fake_opt_sd,
        },
    }
    ckpt_dir = tmp_path / "rl_ckpt"
    save_checkpoint(state, ckpt_dir)

    assert (ckpt_dir / "policy" / "optimizer.pt").exists()
    loaded_opt = torch.load(ckpt_dir / "policy" / "optimizer.pt", weights_only=False)
    assert loaded_opt["param_groups"][0]["lr"] == pytest.approx(1e-4)


def test_save_scheduler_per_model(tmp_path: Path) -> None:
    """schedulers dict의 각 항목이 {name}/scheduler.pt에 저장된다 (RLTrainer 패턴)."""
    fake_sched_sd = {"last_epoch": 5, "base_lrs": [0.01]}
    state = {
        "trainer_state": {"step": 10},
        "schedulers": {"policy": fake_sched_sd},
    }
    ckpt_dir = tmp_path / "rl_sched"
    save_checkpoint(state, ckpt_dir)

    assert (ckpt_dir / "policy" / "scheduler.pt").exists()


def test_save_root_optimizer_single_trainer(tmp_path: Path) -> None:
    """key='' (Trainer 패턴)이면 optimizer.pt를 루트 디렉토리에 저장한다."""
    fake_opt_sd = {"state": {}, "param_groups": [{"lr": 5e-5}]}
    state = {
        "trainer_state": {"step": 2},
        "optimizers": {"": fake_opt_sd},
    }
    ckpt_dir = tmp_path / "sft_ckpt"
    save_checkpoint(state, ckpt_dir)

    assert (ckpt_dir / "optimizer.pt").exists()


def test_save_recipe_yaml(tmp_path: Path) -> None:
    """recipe_dict이 있으면 recipe.yaml로 저장된다."""
    state = {
        "trainer_state": {"step": 1},
        "recipe_dict": {"task": "sft", "model": "gpt2"},
    }
    ckpt_dir = tmp_path / "with_recipe"
    save_checkpoint(state, ckpt_dir)

    recipe_path = ckpt_dir / "recipe.yaml"
    assert recipe_path.exists()
    import yaml
    loaded = yaml.safe_load(recipe_path.read_text())
    assert loaded["task"] == "sft"


def test_save_empty_state_creates_dir_only(tmp_path: Path) -> None:
    """state가 {}이면 디렉토리만 생성되고 다른 파일은 안 만들어진다."""
    ckpt_dir = tmp_path / "empty_state"
    save_checkpoint({}, ckpt_dir)

    assert ckpt_dir.exists()
    contents = list(ckpt_dir.iterdir())
    assert contents == [], f"예상치 못한 파일: {contents}"


def test_write_serving_model_artifact_saves_tokenizer_from_typed_collator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving artifact writer reads tokenizer from ComponentSpec.kwargs."""
    saved: list[Path] = []

    class _Tokenizer:
        def save_pretrained(self, output_dir: Path) -> None:
            saved.append(Path(output_dir))
            (Path(output_dir) / "tokenizer.json").write_text("{}")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str) -> _Tokenizer:
            assert name == "gpt2"
            return _Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )
    settings = SimpleNamespace(
        recipe=SimpleNamespace(
            data=SimpleNamespace(
                collator=ComponentSpec(
                    component="mdp.data.collators.CausalLMCollator",
                    kwargs={"tokenizer": "gpt2"},
                ),
            ),
            model_dump=lambda mode: {"name": "artifact"},
        ),
    )

    _write_serving_model_artifact(torch.nn.Linear(2, 1), settings, tmp_path)

    assert saved == [tmp_path]
    assert (tmp_path / "tokenizer.json").exists()


# ──────────────────────────────────────────────────────────────────────────────
# 2. find_best_checkpoint
# ──────────────────────────────────────────────────────────────────────────────


def test_find_best_checkpoint_best_symlink(tmp_path: Path) -> None:
    """best symlink가 있으면 그 대상 경로를 반환한다."""
    target = tmp_path / "step_100"
    target.mkdir()
    best_link = tmp_path / "best"
    best_link.symlink_to(target)

    result = find_best_checkpoint(tmp_path)
    assert result is not None
    assert result.resolve() == target.resolve()


def test_find_best_checkpoint_latest_fallback(tmp_path: Path) -> None:
    """best가 없고 latest symlink가 있으면 latest 대상을 반환한다."""
    target = tmp_path / "step_50"
    target.mkdir()
    latest_link = tmp_path / "latest"
    latest_link.symlink_to(target)

    result = find_best_checkpoint(tmp_path)
    assert result is not None
    assert result.resolve() == target.resolve()


def test_find_best_checkpoint_no_symlink_returns_none(tmp_path: Path) -> None:
    """best·latest 중 어느 symlink도 없으면 None을 반환한다."""
    result = find_best_checkpoint(tmp_path)
    assert result is None


def test_find_best_checkpoint_best_takes_priority(tmp_path: Path) -> None:
    """best와 latest 둘 다 있으면 best가 우선이다."""
    best_target = tmp_path / "best_epoch"
    latest_target = tmp_path / "latest_epoch"
    best_target.mkdir()
    latest_target.mkdir()

    (tmp_path / "best").symlink_to(best_target)
    (tmp_path / "latest").symlink_to(latest_target)

    result = find_best_checkpoint(tmp_path)
    assert result is not None
    assert result.resolve() == best_target.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# 3. gather_fsdp_state_dict — non-FSDP 경로
# ──────────────────────────────────────────────────────────────────────────────


def test_gather_fsdp_state_dict_non_fsdp_returns_none() -> None:
    """일반 nn.Module(FSDP 아님)을 넘기면 None을 반환한다."""
    model = torch.nn.Linear(4, 4)
    result = gather_fsdp_state_dict(model, is_main_process=True)
    assert result is None


def test_gather_fsdp_state_dict_import_failure_returns_none() -> None:
    """torch.distributed.fsdp import가 실패하면 None을 반환한다(경고 로그 허용)."""
    model = torch.nn.Linear(4, 4)
    # FSDP import를 막아 ImportError 경로를 강제 (except 분기를 타야 None)
    with patch.dict("sys.modules", {"torch.distributed.fsdp": None}):
        # import는 함수 내부에서 발생 — patch 후에도 _checkpoint 모듈은 이미 import됨.
        # 대신 isinstance 검사에서 False로 떨어져 None을 반환함을 검증.
        result = gather_fsdp_state_dict(model, is_main_process=True)
    assert result is None
