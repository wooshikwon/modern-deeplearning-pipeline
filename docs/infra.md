# Infrastructure Guide

MDP run을 실제 GPU/서버 환경에서 돌릴 때의 운영 실전 가이드. 로컬 개발 환경은 `docs/getting-started.md`, 학습 알고리즘·콜백은 `docs/training.md`를 참조한다. 이 문서는 "어떻게 띄워서 어떻게 관찰하고 어떻게 종료하는가"에 초점을 둔다.

## 분산 실행 (torchrun)

`mdp train` / `mdp rl-train`은 `compute.gpus`와 `compute.distributed.strategy` 설정을 읽어 **내부적으로** `_torchrun_entry.py`를 통해 멀티 GPU를 직접 관리한다. 절대 외부 `torchrun`으로 감싸지 말 것 — `torchrun ... mdp rl-train ...` 패턴은 이중 torchrun(`torchrun → MDP → 내부 torchrun`)이 되어 포트 충돌로 실패한다. 올바른 실행은 단순히:

```bash
mdp rl-train -r recipe.yaml -c config.yaml
```

4 GPU DDP 환경에서 이 한 줄만으로 torchrun이 자동 기동되어 rank 0~3에 프로세스가 뜬다. Config의 `compute.distributed.strategy`(`ddp` / `fsdp` / `deepspeed_zero3` / `auto`)가 실제 래퍼를 결정한다.

---

## nohup 환경 모니터링

장시간 run을 백그라운드로 돌리는 가장 일반적 패턴은 nohup + stdout redirect 조합이다:

```bash
nohup mdp rl-train -r recipes/wntp_baseline.yaml -c configs/h200_4gpu.yaml \
  > storage/logs/run_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

이 환경에서 **tqdm은 `sys.stdout.isatty() == False`를 감지해 자동 비활성화**된다. 따라서 진행 상황은 MDP의 text 기반 step progress와 MLflow tracking UI 두 경로로 관찰한다. System Logging(rank-0 filter + 외부 logger WARNING + tqdm 단일화)이 기본 적용되므로 로그가 과거처럼 rank 4배로 부풀지 않는다.

### 기본 관찰 패턴

```bash
# 전체 실시간 tail
tail -f storage/logs/run_baseline_*.log

# step progress와 run 경계만 필터링
tail -f storage/logs/run_baseline_*.log | grep -E "\[step|MDP Run"

# 종료 후 스냅샷으로 훑기
grep -E "MDP Run (Started|Ended)" storage/logs/run_baseline_*.log
```

### 이 조합으로 무엇이 가능한가

- **Start banner**(`MDP Run Started`)로 run이 실제로 학습 루프에 진입했는지 확정. 배너에는 run_id, experiment, algorithm, strategy, precision, max_steps, epochs, per-rank batch size가 포함되어 실행 구성을 한눈에 검증 가능.
- **Step progress**(`[step N/Total | P%] loss=... lr=... grad_norm=... throughput=... ETA=...`)로 진행률, loss 추이, throughput, 예상 완료 시각을 확인. 출력 간격은 recipe의 `monitoring.log_every_n_steps`로 조절한다(기본 10). `986` step run에서 기본값이면 약 98회만 출력되어 로그 파일이 부풀지 않는다.
- **End banner**(`MDP Run Ended`)로 run 종료와 `stopped_reason`(`completed` / `signal_term` / `signal_int` / `early_stopping` / `max_steps` / `oom`) 확인. Duration, total steps, final loss, checkpoints saved, peak memory도 함께 기록.
- **OOM 발생 시 자동 rank별 memory summary**. `torch.cuda.OutOfMemoryError`가 train loop에서 잡히면 `_dump_oom_summary`가 실행되어 `rank 0: allocated=... reserved=... free=...` 표를 정리된 포맷으로 출력하고, `free < 1 GiB`인 rank를 "OOM suspected on rank(s): [...]" 라인에 하이라이트한 뒤 예외를 re-raise한다. 과거처럼 PyTorch traceback 본문에서 수동 파싱할 필요 없음.

### Verbose 모드 on/off

기본 동작이 "조용한 로그"(rank-0 filter + httpx/transformers INFO → WARNING + tqdm 비활성)이므로 대부분의 운영 run에서는 그대로 충분하다. 하지만 외부 라이브러리 내부 동작을 추적해야 하는 디버깅 시에는 verbose 모드로 복원한다:

```bash
# 환경변수 — 한 번만 쓸 때
MDP_LOG_VERBOSE=1 nohup mdp rl-train -r ... -c ... > run.log 2>&1 &
```

또는 recipe에 영구 지정:

```yaml
# recipe.yaml
monitoring:
  verbose: true
```

둘 중 하나라도 true이면 `setup_logging()`이 early-return되어 rank-0 filter·외부 logger downgrade·tqdm 비활성·warning suppression 전부 건너뛴다. 모든 rank의 로그가 다시 보이고 httpx/transformers INFO가 복원된다.

### 진단 옵션 — memory_history snapshot

tensor-level memory 디버깅이 필요한 경우 `monitoring.memory_history`를 활성화한다:

```yaml
# recipe.yaml
monitoring:
  memory_history: true
```

활성 시 run 시작에 `torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="python")`이 호출되고, 종료·OOM 시 `storage/memory_profiles/{run_id}.pickle` 파일이 저장된다(`run_id`는 MLflow active run이 있으면 그 값, 없으면 timestamp fallback). 이 snapshot은 [pytorch.org/memory_viz](https://pytorch.org/memory_viz)에서 로드해 tensor-level allocation을 시각적으로 분석할 수 있다.

- rank-0 + CUDA 가용 조건에서만 활성화됨(multi-rank에서 동일 파일을 덮어쓰는 경합 방지).
- `_record_memory_history` 또는 `_dump_snapshot` 실패는 warning으로 흡수 — 학습 루프를 깨뜨리지 않는다.

로컬 분석 워크플로우:

```bash
# 원격 서버에서 로컬로 pickle 복사
scp server:/path/to/storage/memory_profiles/abc123def456.pickle ./

# 브라우저에서 pytorch.org/memory_viz 열고 파일 업로드
```

### OOM 드릴다운

OOM이 발생해 `_dump_oom_summary` 출력을 받은 경우의 판단 절차:

1. `free < 1 GiB`인 rank가 1개만 있는가 → imbalanced load. DDP micro-batch + activation peak 또는 특정 rank의 데이터 샘플 size outlier 의심. `batch_size`를 낮추거나 gradient accumulation을 늘려 per-rank peak을 줄인다.
2. 모든 rank가 비슷한 상태로 OOM인가 → 모델 자체가 메모리에 안 맞음. `strategy: fsdp` 전환 또는 `precision: bf16`, LoRA adapter 도입, gradient checkpointing 활성화 검토.
3. `memory_history: true`로 다시 돌려 snapshot을 pytorch.org/memory_viz에 올린다 — 어떤 tensor가 peak를 만드는지 레이어 단위로 식별 가능.

---

## Graceful Shutdown

학습 루프가 외부 시그널(Ctrl+C, 상위 `timeout`, K8s eviction 등)로 중단될 때 MLflow run이 zombie(RUNNING) 상태로 남지 않도록, `Trainer.train()` / `RLTrainer.train()`이 SIGTERM·SIGINT를 내부에서 처리한다. 상세한 동작 원칙·분산 안전성은 `docs/training.md` "Graceful Shutdown" 섹션을 참조.

운영 실전 패턴:

```bash
# 2시간 wall-clock 상한 + graceful shutdown
timeout 2h mdp rl-train -r recipe.yaml -c config.yaml \
  > run.log 2>&1
```

- 2시간 후 SIGTERM → 현재 step 완료 후 break → finally(cleanup + on_train_end + MLflow summary) 실행.
- MLflow run은 `FINISHED` 상태로 마감, `stopped_reason=signal_term` tag.
- End banner가 `Stopped: signal_term`으로 rank-0 로그에 남아 grep으로 즉시 식별 가능.

`mdp inference` / `mdp generate` 경로에는 현재 signal handler가 없다 — 장시간 배치 추론에 timeout을 걸면 출력이 부분 저장될 수 있으므로 `--batch-size`와 데이터 분할로 wall-clock을 예측 가능하게 관리한다.

---

## Resume

중단된 run을 최신 체크포인트부터 재개하려면 Config의 `job.resume`을 설정한다:

```yaml
# config.yaml
job:
  resume: auto             # 기본. 최신 체크포인트 자동 감지
  # resume: disabled       # 항상 처음부터
  # resume: ./checkpoints/checkpoint-1000   # 특정 체크포인트
  max_retries: 0           # torchrun 재시작 횟수
```

체크포인트 구조와 mid-epoch resume 동작은 `docs/training.md` "체크포인트 & Resume" 섹션 참조.

분산 환경에서 자동 재시작이 필요하면 `torchrun --max_restarts` 옵션을 내부 torchrun entry로 전달하는 방법은 spec 범위 밖이다 — 현 구조에서는 외부 오케스트레이터(SkyPilot, Slurm, K8s) 레벨에서 재시작을 관리한다.

---

## MLflow Tracking

`config.mlflow.tracking_uri`가 기본값 `./mlruns`이면 run 산출물이 현재 디렉토리에 로컬로 쌓인다. 원격 tracking server(MLflow server 또는 Databricks)를 쓰려면:

```yaml
# config.yaml
mlflow:
  tracking_uri: http://mlflow.example.com:5000
  experiment_name: weighted-ntp-baseline
```

MLflow 자체 통신 실패는 학습을 깨뜨리지 않는다 — `log_step_metrics`의 매 step 호출은 `logger.debug`로 에러를 흡수하고, 1회성 호출(`log_static_params`, `log_summary`)은 `logger.warning`로 경고만 낸다. 상세는 `docs/training.md` "MLflow Logging Conventions" 섹션.

---

## 에러 복구 전략

MDP는 프로세스 내 복구를 시도하지 않는다. 대신:

1. `ModelCheckpoint` 콜백으로 주기적 저장
2. `job.resume: auto`로 최신 체크포인트에서 재시작
3. MLflow/monitoring 실패는 경고만 출력하고 학습은 계속
4. SIGTERM/SIGINT는 Graceful Shutdown 경로로 처리되어 `stopped_reason=signal_term|signal_int` tag와 함께 정상 마감
5. OOM은 `_dump_oom_summary` 실행 후 `stopped_reason=oom` tag 기록, 재시작은 운영자가 판단(batch size 축소 등)

원격·클라우드 오케스트레이션(SSH job 제출, SkyPilot 런칭, K8s scheduling)은 MDP의 책임이 아니다. 사용자가 이미 실행 환경(로컬 머신 / SSH로 접속한 원격 서버 / 클라우드 컨테이너) 안에 있다고 가정한다. 원격 실행 자동화가 필요하면 SkyPilot, Ray, SLURM, K8s 등 전용 오케스트레이터를 쓴다.
