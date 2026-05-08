"""torchrun 엔트리포인트 -- 분산 학습 워커가 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json
import logging
import os


def _init_distributed_if_torchrun(run_plan) -> None:
    """torchrun 워커로 실행 중이면 process group을 초기화한다."""
    from mdp.runtime.worker import init_distributed_if_torchrun

    init_distributed_if_torchrun(run_plan)


def run_training(run_plan) -> dict:
    """Worker adapter for the runtime-owned SFT/RL training lifecycle."""
    from mdp.cli.callback_output import print_callbacks_log
    from mdp.runtime.training import run_training as runtime_run_training

    return runtime_run_training(
        run_plan,
        callbacks_observer=print_callbacks_log,
    )


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
        "--run-plan-path", required=True, help="RunPlan payload JSON 파일 경로"
    )
    parser.add_argument(
        "--result-path", default=None, help="학습 결과를 저장할 JSON 파일 경로 (rank-0 전용)"
    )
    args = parser.parse_args()

    from mdp.runtime.payload import RunPlanPayload

    with open(args.run_plan_path) as f:
        raw = json.load(f)

    run_plan = RunPlanPayload.from_json_dict(raw).to_run_plan()
    settings = run_plan.settings

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
    _init_distributed_if_torchrun(run_plan)

    # Liger monkey-patch와 materialization은 run_training() 내부 ExecutionEngine
    # 경로에서 수행된다. 여기서는 dist init 선행만 보장한다.
    result = run_training(run_plan)

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
