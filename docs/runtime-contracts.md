# Runtime Contracts

This document captures cross-cutting contracts shared by training, inference,
serving, checkpointing, and configuration validation.

## Validation Scope

| Command | Validation scope |
|---|---|
| `mdp train`, `mdp rl-train` | Full Recipe/Config validation: task fields, heads, adapters, data, distributed, component imports, runtime compatibility. |
| `mdp estimate` | Model and configuration shape needed for memory estimation. |
| `mdp inference --run-id/--model-dir` | Model-related compatibility and artifact loading. |
| `mdp inference --pretrained`, `mdp generate --pretrained` | Recipe-less path; invalid model/runtime combinations fail at load/runtime. |

Recipe uses `extra="forbid"`. A legacy `callbacks:` block in Recipe is invalid.

## Model Source Plan

CLI source flags are resolved once into `ModelSourcePlan`.

- `kind`: `artifact` or `pretrained`.
- `command`: `inference`, `generate`, `serve`, or `export`.
- Artifact source: resolved path.
- Pretrained source: URI string.

The plan enforces mutual exclusion and command support before loaders run.

## Checkpoint Manifest

New checkpoints contain `manifest.json` with:

- `layout_version`
- `kind`: `sft` or `rl`
- `global_step`, `epoch`, `step_in_epoch`, `saved_at`
- trainer state path
- optional recipe/config snapshot paths
- optional scaler path
- model records keyed by slot name

Each model record declares role, weight format, relative path, trainable flag,
and optional optimizer/scheduler paths.

Manifestless checkpoints are legacy. They are read best-effort through
`trainer_state.json`, `scaler.pt`, and older filename conventions.

## Strategy Capability

Strategies expose checkpoint capability declaratively.

| Strategy | Managed checkpoint | Save participation | Status |
|---|---:|---|---|
| none | yes | main process | supported |
| DDP | yes | rank 0 writes full state | supported |
| FSDP | yes | all ranks enter collective; rank 0 writes files | supported |
| DeepSpeed ZeRO | no | engine-owned | unsupported, fail-fast |

DeepSpeed aliases may appear in catalogs for forward compatibility, but
Trainer/RLTrainer reject them until an engine-contract implementation owns
backward, optimizer step, ZeRO shards, and checkpoint semantics.

## Precision And Device Map

- `bf16` requires Ampere-class GPU or newer.
- `flash_attention_2` requires compatible GPU and installed package support.
- `device_map` is inference/serving-only. Training rejects models already
  distributed by HuggingFace device maps.
- FSDP + QLoRA is currently incompatible. Use DDP, or wait for a separate
  DeepSpeed engine-contract path.

## Forward Contract

Model forward/training outputs must be normalized before shared trainer logic
consumes them. RL algorithms declare their forward needs through contract flags
so the trainer can avoid unnecessary logits/materialization when hidden-state or
fused-loss paths are sufficient.

## Runtime Precedence

- Generation kwargs: `MDP defaults < Recipe generation < explicit CLI/serving args`.
- Serving config: `artifact config snapshot < explicit serve CLI`.
- Callback source: only `--callbacks callbacks.yaml`; Recipe has no callback list.
