# MDP Architecture

High-level architecture overview, component layout, and key design decisions. For usage patterns, see `AGENT.md`. For training-specific details, see `docs/training.md`.

## Directory Layout

```
mdp/
├── aliases.yaml          # _component_ short name -> full path mapping
├── callbacks/            # Shared callback base classes (training + serving)
│   ├── base.py           # BaseCallback, BaseInferenceCallback, BaseInterventionCallback
│   ├── inference.py      # DefaultOutputCallback
│   └── interventions/    # ResidualAdd, LogitBias
├── cli/                  # CLI entry points (typer)
├── data/                 # Data pipeline (Dataset, DataLoader, transforms)
├── assembly/             # Component assembly planning/materialization
│   ├── assembly_plan.py  # AssemblyPlan graph
│   ├── planner.py        # RunPlan -> AssemblyPlan
│   ├── materializer.py   # AssemblyPlan -> concrete training bundles/components
│   ├── bundles.py        # Materialized training bundle dataclasses
│   └── specs.py          # Assembly node/spec dataclasses
├── models/               # Model layer
│   ├── base.py           # BaseModel ABC
│   ├── pretrained.py     # PretrainedResolver (hf://, timm://, ...)
│   ├── family_routing.py # Semantic name → actual module name translation
│   ├── heads/            # Task-specific output heads
│   ├── adapters/         # LoRA, QLoRA, PrefixTuning
│   └── catalog/          # YAML model metadata
├── monitoring/           # Data distribution monitoring
├── artifacts/            # Artifact layout descriptors, weight loading, serving writers
├── serving/              # Serving layer (FastAPI)
├── runtime/              # Training runtime launcher/worker/engine
├── settings/             # Settings loading, schema validation, and RunPlan building
├── training/             # Training engine (see below)
└── utils/
    ├── estimator.py      # GPU memory estimator
    ├── sanitize.py       # Config masking
    └── logging.py        # System logging setup (setup_logging, Rank0Filter)
```

## Runtime Control Plane

Training execution is explicit across three planning/execution objects:

```
Raw YAML / artifact snapshot / CLI args
  -> SettingsLoader
  -> Settings
  -> RunPlanBuilder
  -> RunPlan
  -> AssemblyPlan
  -> ExecutionEngine
```

`SettingsLoader` owns source loading only: it reads recipe/config/artifact
sources, applies overrides, performs environment substitution, and returns a
validated `Settings` object. It does not wrap runtime planning.

`RunPlanBuilder` combines the loaded `Settings` with command intent, validation
scope, source paths, callback config path, artifact source, and distributed
intent to produce the `RunPlan`. `RunPlan` is the runtime source of truth after
settings loading has completed.

`AssemblyPlanner` converts the validated `RunPlan` into an
`AssemblyPlan`, a serializable graph of model, data, strategy, callback, and
trainer nodes. `AssemblyPlan` does not hold live model, optimizer, dataloader,
or callback instances.

`AssemblyMaterializer` turns the graph into concrete bundles after worker-side
runtime setup has completed. `ExecutionEngine` owns the final SFT/RL dispatch:
it builds the assembly plan, materializes callbacks and bundles, then invokes
`Trainer.from_bundle(...).train()` or `RLTrainer.from_bundle(...).train()`.

`mdp/runtime/__init__.py` intentionally performs no eager imports. Consumers
must import concrete runtime modules directly so importing `mdp.runtime` never
pulls in the training stack as a side effect.

### Runtime modules

```
runtime/
├── __init__.py     # no eager imports
├── launcher.py     # parent-process single/torchrun launch decision
├── worker.py       # worker-side setup order
├── context.py      # RuntimeContext
└── engine.py       # ExecutionEngine
```

## training/ Internal Layers

`mdp/training/` is structured as three layers stacked beneath the public trainer classes.

```
training/
├── trainer.py          # Trainer(BaseTrainer)  — SFT epoch loop
├── rl_trainer.py       # RLTrainer(BaseTrainer) — RL step loop
│
├── _base.py            # BaseTrainer(ABC) — shared lifecycle + observability wrappers
├── _checkpoint.py      # CheckpointManager + manifest-aware checkpoint I/O
├── _features.py        # Feature extractor dispatcher free functions
│
├── _progress_log.py    # Python logger + stdout helpers (renamed from _logging_helpers.py)
├── _mlflow_logging.py  # MLflow logging helpers
├── _schedulers.py      # Warmup scheduler assembly helpers
├── _common.py          # Shared utilities (detect_device, backward_and_step, ...)
│
├── callbacks/          # EarlyStopping, ModelCheckpoint, EMA + re-export base
├── losses/             # DPOLoss, GRPOLoss, PPOLoss + BaseAlgorithm + _ce_helpers.py
└── strategies/         # DDPStrategy, FSDPStrategy, DeepSpeedStrategy stub (unsupported)
```

### BaseTrainer (`_base.py`)

`class BaseTrainer(ABC)` is the common abstract base for `Trainer` and `RLTrainer`. It consolidates lifecycle infrastructure that was previously duplicated in both trainers.

**Responsibilities:**
- Common wrapper methods: `_fire`, `_move_to_device`, `_should_stop`, `_estimate_total_steps`
- OOM/memory_history wrappers: `_dump_oom_summary`, `_maybe_start_memory_history`, `_maybe_dump_memory_snapshot` (delegate to `_progress_log` free functions)
- System logging wrappers: `_fmt_eta`, `_log_step_progress`, `_log_run_banner` (LR lookup delegated to abstract `_optimizer_for_progress_log()`)
- MLflow lifecycle wrappers: `_start_mlflow_run`, `_log_mlflow_params`, `_log_mlflow_summary`, `_peak_memory_summary_extra` (subclasses implement `_collect_mlflow_params()`)
- Checkpoint state hooks: `abstractmethod _checkpoint_state() -> dict`, `_load_checkpoint_state(state: dict)` — consumed by `_checkpoint.py` call sites

**What Trainer and RLTrainer keep:**
- `__init__` (component assembly)
- `train()` main loop (epoch/step loop, callback dispatch, error handling)
- Step execution methods (`_train_step_*`, `_forward_preference`, etc.)
- Validation loops (`_validate`, `_run_rl_validation`, etc.)

### `_features.py` — Feature Extractor Dispatcher

Stateless free functions for extracting `(hidden_states, head_weight)` from models. No dependency on trainer state.

| Function | Purpose |
|---|---|
| `extract_hidden_states_and_head(model, batch, layer_idx)` | Framework dispatcher — routes to HF / timm / torchvision |
| `_extract_hf_pretrained(model, batch, layer_idx)` | HuggingFace pretrained model path |
| `_extract_timm(model, batch)` | timm model path |
| `_extract_torchvision_resnet(model, batch)` | torchvision ResNet path |
| `extract_logits(model_output)` | Logits extraction from model output |
| `forward_model(model, batch, role)` | Standardized forward pass with role-keyed output |

`RLTrainer` calls these functions directly. Thin bound-method wrappers remain as
part of the stable low-level trainer surface for callers that exercise trainer
internals directly.

### `_checkpoint.py` — Checkpoint I/O

Manifest-aware checkpoint I/O for training state. Separated from the compute layer (trainer loop) to isolate side effects.

| Function | Purpose |
|---|---|
| `CheckpointManager.save(context, slots, strategy, scaler)` | Write manifest, model weights, optimizer/scheduler/scaler, and trainer state |
| `CheckpointManager.load(ckpt_dir)` | Read manifest checkpoints; bounded reader for manifestless checkpoints |
| `save_checkpoint(state, ckpt_dir)` / `load_checkpoint(ckpt_dir)` | Stable trainer-facing wrapper API over `CheckpointManager` |
| `gather_fsdp_state_dict(model)` | Collect full state dict via all-rank FSDP collective |
| `find_best_checkpoint(strategy_config)` | Resolve best/latest symlink to checkpoint path |

Written checkpoints contain `manifest.json` with `layout_version`, checkpoint
kind, trainer state file, optional recipe/config snapshots, scaler path, and
per-model records. Each model record declares role, weight format, relative
path, trainable flag, and optional optimizer/scheduler files. Manifestless
directories are a read-only compatibility boundary: `CheckpointManager.load()`
can read them best-effort from `trainer_state.json` and `scaler.pt`, while
checkpoint writes use the manifest layout.

Model record paths preserve file/directory meaning. A file record such as
`policy/model.safetensors` uses its parent directory as the weight root; a
`pretrained_dir` record such as `policy` uses that directory itself as the
weight root. This keeps SFT root checkpoints, RL named slots, and HF
`save_pretrained()` sharded directories under one loader contract.

Strategies expose `checkpoint_capability`. DDP and FSDP opt in to manager-owned checkpointing; FSDP also declares that save is an all-rank collective. For PEFT/LoRA under non-collective strategies such as DDP, `CheckpointManager` uses the strategy `unwrap()` read-only contract to detect the adapter model and writes the adapter `save_pretrained()` layout instead of copying frozen base weights into every checkpoint. The default strategy capability is unsupported, so DeepSpeed is intentionally fail-fast in the current Trainer/RLTrainer runtime. Its engine owns backward, optimizer step, ZeRO shards, and checkpoint semantics, so DeepSpeed ZeRO checkpoints must not be documented or restored as normal DDP/FSDP checkpoints until a separate engine-contract spec implements that path.

`Trainer` and `RLTrainer` call these via `BaseTrainer._checkpoint_state()` / `_load_checkpoint_state()` hooks — they no longer own checkpoint save/restore logic directly. MLflow model snapshots are not written by `_checkpoint.py`; trainers delegate serving artifact construction to `ServingArtifactManager(mode="mlflow_snapshot")`.

### `artifacts/` — Layout, Loading, and Serving Writers

`mdp.artifacts` is the neutral package shared by training and serving. It keeps
filesystem layout observation separate from lifecycle-specific policy.

| Module | Purpose |
|---|---|
| `mdp.artifacts.layout` | Filename constants, `WeightLayout`, directory layout detection, adapter-name detection |
| `mdp.artifacts.loading` | Load `WeightLayout` into models, including full-state, safetensors, HF sharded directories, and PEFT adapters |
| `mdp.artifacts.serving` | `ServingArtifactManager` write modes: `mlflow_snapshot`, `deployment_export`, `custom_export` |

MLflow snapshot and deployment export intentionally have different semantics.
The snapshot records the training artifact as observed, so adapter runs stay
adapter-only. Deployment export builds a serving package and may merge adapters
when the requested model/source supports it.

### `_progress_log.py` — Progress Logging

Renamed from `_logging_helpers.py` (spec-training-restructure U2) to distinguish from `utils/logging.py` (system logging infrastructure). Contains the 6 free functions for trainer progress output:

`fmt_eta`, `log_step_progress`, `log_run_banner`, `dump_oom_summary`, `maybe_start_memory_history`, `maybe_dump_memory_snapshot`

`BaseTrainer` wraps these as bound methods so subclasses inherit them without re-importing.

### Inheritance diagram

```
BaseTrainer(ABC)
├── _base.py
│   ├── wrapper methods (delegating to _progress_log + _mlflow_logging)
│   └── abstract hooks (_checkpoint_state, _collect_mlflow_params, _optimizer_for_progress_log)
│
├── Trainer(BaseTrainer)          — trainer.py
│   ├── train() — epoch loop
│   ├── _validate()
│   ├── _checkpoint_state() → save via _checkpoint.save_checkpoint()
│   └── _collect_mlflow_params()
│
└── RLTrainer(BaseTrainer)        — rl_trainer.py
    ├── train() — step loop
    ├── _train_step_offline() / _train_step_generation()
    ├── _checkpoint_state() → save via _checkpoint.save_checkpoint()
    └── _collect_mlflow_params()
```

## 3-Tier Component Architecture

```
Tier 3 (Orchestrator)
  CLI commands, runtime launcher/worker, and ExecutionEngine
  Load RunPlan, launch workers, and dispatch training bundles or serving paths
  ↓
Tier 2 (Composite)
  SettingsLoader, RunPlanBuilder, AssemblyPlanner, AssemblyMaterializer
  AssemblyMaterializer        — stable component creation API, singleton caching
  Trainer/RLTrainer — training loop execution
  (AssemblyMaterializer and Trainer are both Tier 2 but do not create each other)
  ↓
Tier 1 (Atomic)
  Dataset, Transform, Optimizer, Scheduler, Loss, Head, Callback, Evaluator
```

Import direction flows top-to-bottom only. `import-linter` enforces the stable
module boundaries in CI. `mdp.runtime.engine` is allowed to import
`mdp.training` because it owns training dispatch. `mdp.serving` and
`mdp.settings` must not import `mdp.runtime.engine`; serving/inference paths
must not accidentally depend on the training execution engine.

## Key Design Decisions

**`_component_` pattern**: Pluggable slots use `_component_: <alias or
full.path>` in YAML. MDP-owned settings stay in closed schema blocks, and
component-specific kwargs remain open only inside typed component envelopes.
Runtime code consumes typed specs and materializer plan specs rather than raw
`_component_` dictionaries.

**Recipe / Config / Callbacks separation**: Recipe = what to train, Config = where to run it, Callbacks = side-channel observation/intervention. `--callbacks <yaml>` is the only callback injection path; `Recipe` has no `callbacks:` field.

**BaseTrainer inheritance over mixin**: `Trainer` and `RLTrainer` share a common lifecycle (init → mlflow start → epoch/step loop → checkpoint → mlflow end). A single abstract base class captures this cleanly; mixins would add MRO complexity.

**`_`-prefix = private to owning namespace**: `_base.py`, `_checkpoint.py`, `_features.py`, `_progress_log.py`, `_mlflow_logging.py`, `_schedulers.py`, `_common.py` are internal implementation modules. CLI entrypoints may import selected private helpers to bridge command-line YAML into runtime objects; serving and external consumer packages should not depend on training-private modules. Public trainer entrypoints remain `Trainer` and `RLTrainer`.

**`_features.py` is stateless**: Feature extractor dispatcher depends only on `(model, batch, layer_idx)` — no trainer state. This enables reuse from inference callbacks without instantiating a trainer.

**`losses/_ce_helpers.py` is stateless**: Sibling module to `_features.py`. While `_features.py` dispatches hidden state extraction, `_ce_helpers.py` provides the `compute_per_token_ce_chunked_from_hidden` free function for memory-efficient CE computation from hidden states. Both modules follow the same stateless free function principle (spec-training-restructure principle 4): no dependency on trainer or algorithm instance state, `chunk_size` passed as an explicit parameter so callers retain full control.

**Checkpoint I/O is a separate layer**: `_checkpoint.py` free functions are pure I/O. The trainer loop knows *when* to checkpoint; `_checkpoint.py` knows *how*. DDP/FSDP edge cases (collective state_dict gather, rank-0-only save) are isolated here.
