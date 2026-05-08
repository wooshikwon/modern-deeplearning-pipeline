"""torchrun 엔트리포인트 -- 분산 학습 워커가 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json
import logging
import os


def _init_distributed_if_torchrun(settings) -> None:
    """torchrun 워커로 실행 중이면 process group을 초기화한다."""
    from mdp.runtime.worker import init_distributed_if_torchrun

    init_distributed_if_torchrun(settings)


def _is_main_process() -> bool:
    """rank-0 여부를 반환한다. torchrun 환경이 아니면 항상 True."""
    from mdp.runtime.worker import is_main_process

    return is_main_process()


def _print_callbacks_log(
    cb_instances: list,
    settings,
) -> None:
    """실행 시작 시 적용된 callbacks와 training.* 자동 추가 항목을 출력한다.

    is_json_mode()이면 생략. rank-0에서만 출력.
    """
    from mdp.cli.output import is_json_mode

    if is_json_mode() or not _is_main_process():
        return

    from mdp.callbacks.base import BaseInterventionCallback

    lines = []
    for cb in cb_instances:
        tag = "[Int]" if isinstance(cb, BaseInterventionCallback) else "[Obs]"
        cb_name = type(cb).__name__
        # 대표 속성 표시 (monitor 있으면)
        attrs = []
        if hasattr(cb, "monitor"):
            attrs.append(f"monitor={cb.monitor}")
        detail = f" ({', '.join(attrs)})" if attrs else ""
        lines.append(f"  {tag} {cb_name}{detail}")

    # training.* 자동 추가 항목
    auto_lines = []
    training = settings.recipe.training
    if training.early_stopping is not None:
        es = training.early_stopping
        auto_lines.append(
            f"  EarlyStopping (monitor={es.monitor}, patience={es.patience})"
        )
    if training.ema is not None:
        ema = training.ema
        auto_lines.append(f"  EMA (decay={ema.decay})")

    if lines or auto_lines:
        if lines:
            print("Applied callbacks:")
            for line in lines:
                print(line)
        if auto_lines:
            print("Auto-promoted from training.*:")
            for line in auto_lines:
                print(line)


def run_training(settings, cb_configs: list[dict] | None = None) -> dict:
    """Settings 객체를 받아 ExecutionEngine으로 training lifecycle을 실행한다."""
    # Liger-Kernel monkey-patch는 HF `from_pretrained`(Factory.create_model 내부)
    # 이전에 수행되어야 효과가 있다. Engine이 AssemblyMaterializer를 호출하기
    # 직전에 공통 적용해 single/torchrun worker 양쪽 순서를 맞춘다. Idempotent.
    # 상세: mdp/_liger_patch.py, spec-algorithm-hidden-states-support §U2.
    from mdp.runtime.context import training_settings_plan_from_settings
    from mdp.runtime.engine import ExecutionEngine
    from mdp.runtime.worker import apply_liger_patches_for_training

    apply_liger_patches_for_training()

    settings_plan = training_settings_plan_from_settings(
        settings,
        cb_configs=cb_configs,
    )
    return ExecutionEngine(callbacks_observer=_print_callbacks_log).run(settings_plan)


def run_rl_training(settings, cb_configs: list[dict] | None = None) -> dict:
    """Compatibility wrapper for callers that still enter through RL training."""
    from mdp.runtime.context import training_settings_plan_from_settings
    from mdp.runtime.engine import ExecutionEngine
    from mdp.runtime.worker import apply_liger_patches_for_training

    apply_liger_patches_for_training()

    settings_plan = training_settings_plan_from_settings(
        settings,
        command="rl-train",
        cb_configs=cb_configs,
    )
    return ExecutionEngine(callbacks_observer=_print_callbacks_log).run(settings_plan)


def main() -> None:
    # PYTORCH_CUDA_ALLOC_CONF must be set before any CUDA allocation (including
    # dist.init_process_group(nccl)). expandable_segments:True lets the caching
    # allocator grow existing memory segments instead of creating new fixed-size
    # blocks, eliminating fragmentation-driven OOM in FSDP training.
    # setdefault preserves any value the caller may have already exported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 워커 프로세스의 기본 로깅 레벨은 WARNING이라 logger.info()가 묻힌다.
    # INFO로 설정하여 GC/FSDP 진단 메시지가 로그에 남도록 한다.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # spec-system-logging-cleanup §U2: Rank-0 filter · 외부 logger downgrade ·
    # warning suppress 를 HF ``from_pretrained`` 첫 호출 이전에 완료해야 httpx
    # INFO 요청 로그가 처음부터 차단된다. argparse 이전에 env-only 로 1차
    # setup 을 건다 (settings 가 아직 없으므로 recipe.monitoring.verbose 는
    # 미반영). setup_logging 은 **args-aware idempotent** — 동일 인자 재호출은
    # no-op 이지만, 인자가 바뀌면 이전 Rank0Filter/외부 logger level 을 해제한
    # 뒤 새 인자로 재조립한다. 따라서 settings 로드 후 2차 호출이 recipe
    # `monitoring.verbose=true` 를 실제로 발효시킬 수 있다.
    from mdp.cli._logging_bootstrap import bootstrap_logging
    bootstrap_logging()

    parser = argparse.ArgumentParser(description="MDP torchrun worker")
    parser.add_argument(
        "--settings-path", required=True, help="Settings JSON 파일 경로"
    )
    parser.add_argument(
        "--result-path", default=None, help="학습 결과를 저장할 JSON 파일 경로 (rank-0 전용)"
    )
    args = parser.parse_args()

    from mdp.settings.schema import Settings

    with open(args.settings_path) as f:
        raw = json.load(f)

    # __cb_configs는 Settings 스키마 밖 임시 필드로 전달됨 (extra="forbid" 우회)
    cb_configs: list[dict] | None = raw.pop("__cb_configs", None)

    settings = Settings(**raw)

    # settings 가 준비된 뒤 bootstrap_logging(settings) 재호출 — recipe
    # monitoring.verbose 값과 env 가 OR 합성된다. setup_logging 의 args-aware
    # idempotency 가 1차/2차 인자 차이를 감지해 Rank0Filter 제거 + 외부 logger
    # level 복원으로 verbose 모드 전환을 실제로 수행한다. **이 2차 호출 제거
    # 금지** — 제거 시 recipe.verbose=True 가 무력화되어 디버깅 recipe 가
    # silent 하게 운영 기본값으로 돌아간다 (cycle 1 review 1-2 의 근본 문제).
    bootstrap_logging(settings)

    # Worker-side order: logging bootstrap → settings load →
    # bootstrap_logging(settings) → dist init → Liger patch(run_training) →
    # assembly materialization(ExecutionEngine).
    _init_distributed_if_torchrun(settings)

    # Liger monkey-patch와 materialization은 run_training() 내부 ExecutionEngine
    # 경로에서 수행된다. 여기서는 dist init 선행만 보장한다.
    result = run_training(settings, cb_configs=cb_configs)

    # rank-0만 결과를 저장한다
    if args.result_path and result:
        try:
            import torch.distributed as dist
            is_main = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            is_main = True
        if is_main:
            with open(args.result_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
