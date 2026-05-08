# Runtime Contracts

This document captures cross-cutting contracts shared by training, inference,
serving, checkpointing, and configuration validation.

## Validation Scope

| Command | Validation scope |
|---|---|
| `mdp train`, `mdp rl-train` | `training`: Recipe/Config schema validation plus business/runtime compatibility: task name, head/task compatibility, adapter constraints, distributed compatibility. Unknown task names are errors. Registered aliases and MDP-owned component import failures are errors; unregistered catalog/custom import paths may warn during validation and still fail later if the active runtime instantiates them. |
| `mdp estimate` | `estimation`: Model and configuration shape needed for memory estimation. Unknown task names may warn instead of failing. |
| `mdp inference --run-id/--model-dir` | `artifact` or `recipe`: Model-related compatibility and artifact loading. |
| `mdp inference --pretrained`, `mdp generate --pretrained` | Recipe-less path; invalid model/runtime combinations fail at load/runtime. |

Recipe and Config use closed Pydantic models for MDP-owned settings. A Recipe
`callbacks:` block is invalid; callbacks are loaded only from the CLI
`--callbacks` file. Component kwargs remain open only inside typed
`_component_` envelopes.

The YAML loader rejects duplicate keys, empty files, and wrong root types before
Pydantic validation. Error messages include the source path and YAML dot-path.

Schema validation errors include source file and YAML dot-path context. In JSON
mode, `mdp train` and `mdp rl-train` keep the same human-readable error message
and also expose machine-readable paths:

```json
{
  "status": "error",
  "command": "train",
  "error": {
    "type": "ValidationError",
    "message": "...",
    "details": {
      "schema_errors": [
        {"path": "$.training.val_check_units", "message": "Extra inputs are not permitted"}
      ]
    }
  }
}
```

## Training Runtime Control Plane

Training runtime setup is represented as an explicit plan-to-execution chain:

```
Raw YAML / artifact snapshot / CLI args
  -> SettingsLoader
  -> Settings
  -> RunPlanBuilder
  -> RunPlan
  -> AssemblyPlanner
  -> AssemblyPlan
  -> ExecutionEngine
```

`SettingsLoader` owns source loading only: it reads recipe/config/artifact
sources, applies overrides, performs environment substitution, and returns a
validated `Settings` object.

`RunPlanBuilder` combines `Settings` with command/runtime metadata to create
`RunPlan`, the validated command intent. It contains validation scope,
command/mode, source paths, override list, callback configs, artifact source,
and distributed intent. It does not preserve raw YAML dictionaries as the
runtime source of truth.

`AssemblyPlan` is the component graph derived from `RunPlan`. It records
model roles, data, trainer kind, strategy, and callbacks as node/spec objects,
not live Python component instances. This keeps process-group initialization,
Liger patching, and device setup ahead of model/dataloader materialization in
torchrun workers.

`ExecutionEngine` owns SFT/RL dispatch for training. It builds the assembly
plan, materializes callbacks and training bundles, then invokes
`Trainer.from_bundle(...).train()` or `RLTrainer.from_bundle(...).train()`.
`SettingsLoader.load_training_settings()` is a source-loading API that returns
validated `Settings`; it does not wrap runtime planning. `RunPlanBuilder`
combines those settings with command/runtime metadata to create the `RunPlan`.
`AssemblyMaterializer(AssemblyPlan).materialize_*` preserves the component
creation/cache API, and direct `Trainer(...)` / `RLTrainer(...)` constructors
remain supported low-level loop APIs for tests and users that inject already
materialized components. The runtime path itself uses `Trainer.from_bundle(...)`
and `RLTrainer.from_bundle(...)`.

`mdp.runtime.training.run_training(...)` is the shared
current-process training helper. It applies Liger patches before
materialization and enters `ExecutionEngine` with the provided validated
`RunPlan`; it does not build a `RunPlan` from `Settings`.

`mdp.cli._torchrun_entry` is a process-entry adapter for torchrun workers.
Its `run_training(...)` wrapper delegates to the runtime helper with the
worker-only callback log observer. There is no separate RL worker wrapper.

Raw `_component_` dictionaries are allowed only at YAML loading/override input,
inside `ComponentSpec.kwargs`, and when serializing snapshots or debug views.
Runtime modules below the settings schema consume `component` and `kwargs`
fields on typed specs or materializer plan specs instead of reading `_component_`
directly.

## Test Boundary Ownership

Runtime contracts are tested at the boundary that owns the behavior:

- `tests/contract` protects public-internal runtime boundaries:
  `RunPlan`, `LaunchPlan`, `AssemblyPlan`, `AssemblyMaterializer`,
  `ExecutionEngine`, artifact loading, and checkpoint manifests.
- Unit tests protect pure utilities and small components such as schema
  validation, family routing, samplers, callback YAML parsing, and algorithm
  math.
- CLI/e2e tests protect the real command surface: `python -m mdp`, `--format
  json`, stdout/stderr separation, `mdp init`, `mdp list`, and command-specific
  error payloads.
- Cloud suites protect hardware boundaries: CUDA, NCCL, prepared fixtures, and
  memory ceilings. They are not the owner for CPU-local runtime contract smoke.

Tests in `tests/contract` should not pin private method names or internal call
order. They should assert stable outputs, plan fields, materialized bundle
shape, JSON payloads, and manifest structure.

## Semantic Config Resolution

Model head and adapter configs may use semantic names:

- `head.slot` maps to raw `_target_attr`
- `adapter.target` maps to raw `target_modules`
- `adapter.save` maps to raw `modules_to_save`

`AssemblyPlan` preserves semantic input as authored. The conversion to raw
backend fields happens during `AssemblyMaterializer` model materialization, after
worker setup has selected the process/device context.

Semantic and raw fields are mutually exclusive in the same config. For example,
`target` and `target_modules` together are ambiguous, so materialization fails
instead of silently choosing one. The same rule applies to `save` with
`modules_to_save`, and `slot` with `_target_attr`.

Semantic resolution must not mutate the original `Settings` or `AssemblyPlan`
node config. Materializers create a consumer dict for backend calls while the
plan remains an unchanged record of the validated input contract.

Import boundary:

- `mdp.runtime.engine` may import `mdp.training`; training dispatch is its job.
- `mdp.serving` and `mdp.settings` must not import `mdp.runtime.engine`.
- `mdp.runtime.__init__` performs no eager imports.

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

Recommended checkpoint writes always produce the manifest layout. Manifestless
checkpoint directories are a bounded read-only compatibility boundary; they are
read best-effort through `trainer_state.json`, `scaler.pt`, and older filename
conventions.

## Strategy Capability

Strategies expose checkpoint capability declaratively.

| Strategy | Managed checkpoint | Save participation | Status |
|---|---:|---|---|
| none | yes | main process | supported |
| DDP | yes | rank 0 writes full state | supported |
| FSDP | yes | all ranks enter collective; rank 0 writes files | supported |
| DeepSpeed ZeRO | no | engine-owned | unsupported, fail-fast |

DeepSpeed aliases may appear in catalogs as reserved strategy names, but
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

SFT has two loss owners. If `recipe.loss` is configured, Trainer uses
`loss_fn(logits, labels)` and ignores any native forward loss. If `recipe.loss`
is absent, Trainer reads Tensor `loss` from model forward output.
Raw single-argument vision models are adapted by passing `batch["pixel_values"]`
to `forward(x)`, so timm/torchvision-style classifiers work with `recipe.loss`.

RL does not consume top-level `recipe.loss`. RL algorithms declare their
forward needs through contract flags and own the objective in
`rl.algorithm.compute_loss(trainable_out, frozen_out, batch)`. FSDP with
`needs_hidden_states=True` is rejected until hidden/head extraction is
strategy-aware and can avoid bypassing wrapper forward hooks.

## Runtime Precedence

- Generation kwargs: `MDP defaults < Recipe generation < explicit CLI/serving args`.
- Serving config: `artifact config snapshot < explicit serve CLI`.
- Callback source: only `--callbacks callbacks.yaml`; Recipe has no callback list.
