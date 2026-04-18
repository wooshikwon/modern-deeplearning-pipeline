"""Strategy 조건부 로깅 회귀 테스트 (spec-system-logging-cleanup §U3).

`RLTrainer` FSDP shard baseline 진단 로그는 strategy 종류에 따라 출력 level이
달라져야 한다:

- ``DDPStrategy``: shard 미적용이 설계상 정상이므로 운영 기본 조용 (DEBUG).
  WARNING 레벨 경고(`policy shard 미적용 의심`)가 발생하면 false-positive.
- ``FSDPStrategy`` + actual/expected > 3: shard가 실제로 적용되지 않았다는
  진단이라 WARNING 유지.
- ``FSDPStrategy`` + actual/expected <= 3: 정상 baseline이라 INFO 유지.

`rl_trainer.py::RLTrainer.train()` 내부의 "shard 베이스라인" 블록을 함수 밖에서
직접 실행하기 어려우므로, 이 테스트는 해당 블록과 등가인 strategy-조건부
logging dispatch 로직을 독립 helper에 재구성한다. 구현이 spec의 규칙과 일치하는지
caplog으로 level·메시지를 검증한다. 프로덕션 코드의 실제 분기는 동일한 결정
기준(``type(self.strategy).__name__`` 판별 + ratio 비교)을 사용한다.
"""

from __future__ import annotations

import logging

import pytest


# rl_trainer에서 사용하는 logger 이름과 동일하게 고정 (caplog level 설정 대상)
_TARGET_LOGGER = "mdp.training.rl_trainer"


class _DummyStrategy:
    """테스트용 strategy stub — 클래스명만 의미가 있다."""


class DDPStrategy(_DummyStrategy):  # noqa: D401 — 테스트 stub
    """Mimic the production ``DDPStrategy`` class name."""


class FSDPStrategy(_DummyStrategy):  # noqa: D401 — 테스트 stub
    """Mimic the production ``FSDPStrategy`` class name."""


def _emit_shard_baseline_log(
    strategy: object | None,
    *,
    mem_alloc_gib: float,
    expected_total_gib: float,
    logger_name: str = _TARGET_LOGGER,
) -> None:
    """rl_trainer.py::RLTrainer.train() 내부 shard baseline 블록의 동등 재현.

    프로덕션 코드와 동일한 결정 규칙을 사용한다:

    1. ``type(strategy).__name__`` 기반 분기.
    2. DDP → ``logger.debug``, FSDP + ratio > 3 → ``logger.warning``,
       FSDP + ratio <= 3 → ``logger.info``.
    """
    logger = logging.getLogger(logger_name)
    strategy_name = type(strategy).__name__ if strategy is not None else "NoStrategy"
    is_fsdp = strategy_name.startswith("FSDP")
    is_ddp = strategy_name == "DDPStrategy"

    if is_ddp:
        logger.debug(
            "DDP strategy — no model sharding (expected). "
            "allocated=%.2f GiB",
            mem_alloc_gib,
            extra={"all_ranks": True},
        )
        return

    ratio = mem_alloc_gib / expected_total_gib if expected_total_gib else 0
    if is_fsdp and ratio > 3:
        logger.warning(
            "FSDP shard baseline: actual=%.2f GiB (%.1f×). "
            ">>3× → policy shard 미적용 의심.",
            mem_alloc_gib, ratio,
            extra={"all_ranks": True},
        )
    else:
        logger.info(
            "FSDP shard baseline: actual=%.2f GiB (%.1f×).",
            mem_alloc_gib, ratio,
            extra={"all_ranks": True},
        )


@pytest.fixture
def _caplog_all_levels(caplog):
    """DEBUG 이상 모든 레벨을 포착해 level-conditional 검증을 가능하게 한다."""
    caplog.set_level(logging.DEBUG, logger=_TARGET_LOGGER)
    return caplog


class TestShardBaselineQuietOnDDP:
    """DDPStrategy에서는 WARNING 레벨 경고가 절대 발생하지 않는다."""

    def test_fsdp_shard_baseline_quiet_on_ddp(self, _caplog_all_levels) -> None:
        _emit_shard_baseline_log(
            DDPStrategy(),
            # ratio 로 환산하면 4× — FSDP 였다면 WARNING 이었을 상황.
            mem_alloc_gib=16.0,
            expected_total_gib=4.0,
        )

        warnings = [
            r for r in _caplog_all_levels.records if r.levelno >= logging.WARNING
        ]
        assert warnings == [], (
            f"DDPStrategy 에서 WARNING 이상 로그가 발생했다: "
            f"{[r.getMessage() for r in warnings]}"
        )

        # 그리고 shard 미적용 문구는 어떤 level 에서도 등장하지 않아야 한다 —
        # DDP 모드의 메시지는 중립적인 "no model sharding (expected)" 표현.
        msgs = [r.getMessage() for r in _caplog_all_levels.records]
        assert not any("미적용 의심" in m for m in msgs), (
            f"DDPStrategy 에서 '미적용 의심' 문구가 나왔다: {msgs}"
        )
        assert any("DDP strategy — no model sharding" in m for m in msgs), (
            f"DDPStrategy 에서 중립적 baseline 로그가 없다: {msgs}"
        )


class TestShardBaselineOnFSDP:
    """FSDPStrategy에서는 ratio 기반 조건부 level 이 유지된다."""

    def test_fsdp_shard_baseline_warns_on_fsdp_misconfigured(
        self, _caplog_all_levels
    ) -> None:
        # ratio = 16 / 4 = 4 → >3 이라 WARNING 이 떠야 한다.
        _emit_shard_baseline_log(
            FSDPStrategy(),
            mem_alloc_gib=16.0,
            expected_total_gib=4.0,
        )

        warnings = [
            r for r in _caplog_all_levels.records if r.levelno == logging.WARNING
        ]
        assert len(warnings) >= 1, (
            "FSDPStrategy + ratio > 3 상황에서 WARNING 이 발생하지 않았다: "
            f"{[r.getMessage() for r in _caplog_all_levels.records]}"
        )
        assert any(
            "미적용 의심" in r.getMessage() for r in warnings
        ), f"WARNING 메시지에 '미적용 의심' 문구가 없다: {[r.getMessage() for r in warnings]}"

    def test_fsdp_shard_baseline_info_on_fsdp_normal(
        self, _caplog_all_levels
    ) -> None:
        # ratio = 5 / 4 = 1.25 → <=3 이라 INFO 만 나와야 한다.
        _emit_shard_baseline_log(
            FSDPStrategy(),
            mem_alloc_gib=5.0,
            expected_total_gib=4.0,
        )

        warnings = [
            r for r in _caplog_all_levels.records if r.levelno >= logging.WARNING
        ]
        assert warnings == [], (
            "FSDPStrategy + ratio <= 3 정상 baseline 에서 WARNING 이 발생했다: "
            f"{[r.getMessage() for r in warnings]}"
        )

        infos = [r for r in _caplog_all_levels.records if r.levelno == logging.INFO]
        assert any("FSDP shard baseline" in r.getMessage() for r in infos), (
            f"FSDP 정상 baseline INFO 로그가 없다: {[r.getMessage() for r in infos]}"
        )


class TestAllRanksEscapeHatch:
    """spec §원칙 1 — `extra={"all_ranks": True}` 가 모든 분기에 부착된다."""

    @pytest.mark.parametrize(
        ("strategy", "alloc", "expected"),
        [
            (DDPStrategy(), 16.0, 4.0),       # DDP branch
            (FSDPStrategy(), 16.0, 4.0),      # FSDP + warn branch
            (FSDPStrategy(), 5.0, 4.0),       # FSDP + info branch
        ],
    )
    def test_all_ranks_extra_attached(
        self, _caplog_all_levels, strategy, alloc, expected
    ) -> None:
        _emit_shard_baseline_log(
            strategy, mem_alloc_gib=alloc, expected_total_gib=expected
        )
        assert _caplog_all_levels.records, "아무 로그도 발생하지 않았다"
        for record in _caplog_all_levels.records:
            assert getattr(record, "all_ranks", False) is True, (
                f"record `{record.getMessage()}` 에 all_ranks=True extra 가 없다. "
                "U1 의 Rank0Filter 를 통과하기 위해 필수."
            )


# ── 소스 코드 수준 회귀 테스트 ──
#
# 위 helper 는 구현 스펙을 재현하지만, 실제 프로덕션 파일이 같은 규칙을 따르는지
# AST·텍스트 스캔으로 확인하는 구조적 가드. 블록이 과거 형태(strategy-무관 WARNING)
# 로 회귀하면 실패한다.


def test_rl_trainer_shard_baseline_has_strategy_dispatch() -> None:
    """rl_trainer.py의 shard baseline 블록이 strategy 판별을 수행해야 한다."""
    from pathlib import Path

    src = (
        Path(__file__).parents[2] / "mdp" / "training" / "rl_trainer.py"
    ).read_text()

    # shard baseline 블록 근방에 strategy 판별이 반드시 존재해야 한다.
    assert "type(self.strategy).__name__" in src, (
        "rl_trainer.py 에서 strategy 판별(`type(self.strategy).__name__`)이 사라졌다. "
        "shard baseline 경고의 strategy 조건부화가 풀린 회귀다."
    )
    # DDP 분기가 warning 이 아닌 debug 로 수행되는지 텍스트 수준에서 확인.
    assert "DDP strategy — no model sharding" in src, (
        "DDPStrategy 분기의 중립 메시지가 사라졌다."
    )
    # all_ranks 에스케이프 해치가 부착되어 있어야 한다 (spec §원칙 1).
    assert 'extra={"all_ranks": True}' in src, (
        "shard baseline 로그에 `extra={\"all_ranks\": True}` 가 누락됐다. "
        "U1 의 Rank0Filter 호환 계약 위반."
    )


def test_rl_trainer_shard_baseline_no_unconditional_warning() -> None:
    """strategy 무관하게 일률적으로 WARNING 을 내던 과거 형태가 없어야 한다."""
    from pathlib import Path

    src = (
        Path(__file__).parents[2] / "mdp" / "training" / "rl_trainer.py"
    ).read_text()

    # 과거 코드: `if _rank == 0:` 바로 밑에서 조건 분기 없이 `logger.warning("...미적용 의심...")`
    # 을 뿜던 형태. 현재는 `_is_fsdp and _ratio > 3` 가드 뒤에서만 warning 이 나와야 한다.
    # 간접 검증: LoRA freeze warning 도 FSDP 가드 하에 있는지 (`_is_fsdp` 토큰 존재).
    assert "_is_fsdp" in src, (
        "FSDP 여부 판별 변수(`_is_fsdp`) 가 사라졌다. "
        "LoRA freeze warning 이 DDP 환경에서도 뜨는 false-positive 가능."
    )
