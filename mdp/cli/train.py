"""mdp train -- 모델 학습을 실행한다."""

from __future__ import annotations

import logging

import typer

from mdp.cli.output import (
    build_error,
    build_result,
    emit_result,
    is_json_mode,
    schema_error_details_from_message,
)

logger = logging.getLogger(__name__)


def _detect_gpu_count() -> int:
    """사용 가능한 GPU 수를 반환한다."""
    from mdp.runtime.launcher import detect_gpu_count

    return detect_gpu_count()


def _run_single(run_plan, callbacks_observer=None) -> dict:
    """단일 GPU/CPU 학습을 실행한다."""
    from mdp.runtime.launcher import run_single

    return run_single(run_plan, callbacks_observer=callbacks_observer)


def _run_distributed(run_plan, nproc: int) -> dict:
    """torchrun을 사용하여 분산 학습을 실행한다."""
    from mdp.runtime.launcher import run_distributed

    return run_distributed(run_plan, nproc)


def run_train(
    recipe_path: str,
    config_path: str,
    overrides: list[str] | None = None,
    callbacks_file: str | None = None,
) -> None:
    """Recipe + Config YAML을 조립하여 학습을 실행한다."""
    from mdp._liger_patch import apply_liger_patches
    from mdp.cli._logging_bootstrap import bootstrap_logging
    from mdp.cli.callback_output import print_callbacks_log
    from mdp.cli.schemas import TrainResult
    from mdp.settings.run_plan_builder import RunPlanBuilder

    # spec-system-logging-cleanup §U2: env-only 1 차 setup. Settings 로드 (HF
    # config 파싱 경유 가능성 있음) · AssemblyMaterializer · HF ``from_pretrained`` 첫 호출
    # 이전에 외부 logger level downgrade 를 걸어두기 위함. setup_logging 은
    # **args-aware idempotent** — 동일 인자는 no-op, 인자가 바뀌면 이전 상태를
    # 해제한 뒤 재조립. settings 로드 후 2차 호출이 recipe `monitoring.verbose`
    # 를 env 와 OR 합성하여 실제로 verbose 모드로 전환한다 (cycle 1 review 1-2).
    bootstrap_logging()

    # Liger monkey-patch는 HF 모델 로드 이전에 적용. 단일 GPU 경로에서는
    # run_training() 내부에서도 한 번 더 호출되지만 idempotent하여 안전하다.
    # 분산 경로에서는 torchrun subprocess의 run_training()이 각 rank에서 적용한다.
    # 상세: mdp/_liger_patch.py, spec-algorithm-hidden-states-support §U2.
    apply_liger_patches()

    if not is_json_mode():
        typer.echo(f"Recipe: {recipe_path}")
        typer.echo(f"Config: {config_path}")
        if callbacks_file:
            typer.echo(f"Callbacks: {callbacks_file}")
        if overrides:
            typer.echo(f"Overrides: {overrides}")

    try:
        run_plan = RunPlanBuilder().training(
            recipe_path,
            config_path,
            overrides=overrides,
            callbacks_file=callbacks_file,
            command="train",
        )
        settings = run_plan.settings

        # recipe monitoring.verbose 를 env 와 OR 합성해 반영. args-aware
        # idempotency 덕에 1차 (env-only) → 2차 (env|recipe) 인자가 달라지면
        # setup_logging 이 Rank0Filter 제거 + 외부 logger level 복원으로
        # verbose 모드로 실제 전환한다. **이 호출 제거 금지** — 제거 시
        # recipe.verbose=True 가 무력화된다.
        bootstrap_logging(settings)

    except Exception as e:
        if is_json_mode():
            message = str(e)
            emit_result(build_error(
                command="train",
                error_type="ValidationError",
                message=message,
                details=schema_error_details_from_message(message),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    from mdp.runtime.launcher import build_launch_plan

    launch_plan = build_launch_plan(
        run_plan,
        nproc=_detect_gpu_count(),
    )
    if not is_json_mode():
        typer.echo(f"GPU count: {launch_plan.nproc}")

    try:
        if launch_plan.distributed:
            if not is_json_mode():
                typer.echo(f"분산 학습 시작 (nproc={launch_plan.nproc})...")
            train_result = _run_distributed(
                run_plan,
                launch_plan.nproc,
            )
        else:
            if not is_json_mode():
                typer.echo("단일 학습 시작...")
            train_result = _run_single(
                run_plan,
                callbacks_observer=print_callbacks_log,
            )

        if not is_json_mode():
            typer.echo("학습 완료.")

        if is_json_mode():
            result = TrainResult(
                checkpoint_dir=settings.config.storage.checkpoint_dir,
                output_dir=settings.config.storage.output_dir,
                metrics=train_result.get("metrics", {}),
                total_epochs=train_result.get("total_epochs"),
                total_steps=train_result.get("total_steps"),
                stopped_reason=train_result.get("stopped_reason"),
                duration_seconds=train_result.get("training_duration_seconds"),
                monitoring=train_result.get("monitoring"),
                run_id=train_result.get("run_id"),
                checkpoints_saved=train_result.get("checkpoints_saved"),
            )
            emit_result(build_result(
                command="train", **result.model_dump(exclude_none=True),
            ))

    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="train",
                error_type="RuntimeError",
                message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] 학습 실패: {e}", err=True)
        logger.exception("학습 실패 상세")
        raise typer.Exit(code=1)
