# Observability

This document groups the output, logging, MLflow, and recovery contracts used by
humans and agents.

## JSON Output

JSON mode writes the result object to stdout.

```json
{
  "status": "success",
  "command": "train",
  "timestamp": "2026-05-06T00:00:00+00:00"
}
```

Errors use:

```json
{
  "status": "error",
  "command": "train",
  "error": {
    "type": "ValidationError",
    "message": "...",
    "details": {}
  }
}
```

## Train Result

Training and RL training results include fields useful to orchestrators:

- `run_id`
- `checkpoint_dir`
- `metrics`
- `total_steps`
- `stopped_reason`
- `checkpoints_saved`
- `duration_seconds`

`checkpoints_saved` is `int | None`. `0` means the run completed but no
checkpoint was written, which usually indicates a monitor/configuration issue.

## stopped_reason

SFT values:

- `completed`
- `early_stopped`
- `max_steps_reached`
- `signal_term`
- `signal_int`
- `oom`

RL values:

- `completed`
- `early_stopping`
- `max_steps`
- `signal_term`
- `signal_int`
- `oom`

## System Logging

`mdp train` and `mdp rl-train` initialize logging automatically. Rank 0 owns
normal progress output. Non-rank-0 logs are suppressed unless verbose logging is
enabled.

`MDP_LOG_VERBOSE=1` enables fuller logs when debugging distributed runs.

## MLflow Conventions

MLflow logging is split into three groups:

- Static parameters: configuration needed to reproduce a run.
- Dynamic metrics: per-step/per-epoch observations.
- Summary tags/metrics: final outcome, checkpoint count, best checkpoint, final
  metrics, duration, total steps.

Trainer and RLTrainer should remain symmetric: when one path logs a lifecycle
event, the other path should expose the analogous event unless the concept is
truly absent.

## Graceful Shutdown

Training installs signal handlers for SIGTERM and SIGINT. At a step boundary the
run exits cleanly, finishes callback cleanup, closes MLflow, and records
`signal_term` or `signal_int`.

Inference and generation currently do not install the training signal handler.
Timeouts can leave partial output files.

## Error Recovery

Agents should prefer structured recovery:

| Error | Likely action |
|---|---|
| Validation/config error | Fix Recipe/Config and retry. |
| CUDA OOM | Reduce batch size, increase gradient accumulation, change precision, or use supported distributed strategy. |
| Distributed launch error | Check `compute.distributed.strategy`, GPU count, and avoid external torchrun. |
| `checkpoints_saved == 0` | Inspect `ModelCheckpoint.monitor`, validation metrics, and `strict`. |
