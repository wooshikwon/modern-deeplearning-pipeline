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
├── factory/              # Component assembly facade
│   └── factory.py        # Factory (caching + delegation)
├── models/               # Model layer
│   ├── base.py           # BaseModel ABC
│   ├── pretrained.py     # PretrainedResolver (hf://, timm://, ...)
│   ├── family_routing.py # Semantic name → actual module name translation
│   ├── heads/            # Task-specific output heads
│   ├── adapters/         # LoRA, QLoRA, PrefixTuning
│   └── catalog/          # YAML model metadata
├── monitoring/           # Data distribution monitoring
├── serving/              # Serving layer (FastAPI)
├── settings/             # Settings system (schema, validation, resolver)
├── training/             # Training engine (see below)
└── utils/
    ├── estimator.py      # GPU memory estimator
    ├── sanitize.py       # Config masking
    └── logging.py        # System logging setup (setup_logging, Rank0Filter)
```

## training/ Internal Layers

`mdp/training/` is structured as three layers stacked beneath the public trainer classes.

```
training/
├── trainer.py          # Trainer(BaseTrainer)  — SFT epoch loop
├── rl_trainer.py       # RLTrainer(BaseTrainer) — RL step loop
│
├── _base.py            # BaseTrainer(ABC) — shared lifecycle + observability shim
├── _checkpoint.py      # Checkpoint I/O free functions
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
- Common shim methods: `_fire`, `_move_to_device`, `_should_stop`, `_estimate_total_steps`
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

`RLTrainer` calls these functions directly and keeps thin bound-method shims for backward compatibility with tests.

### `_checkpoint.py` — Checkpoint I/O

Manifest-aware checkpoint I/O for training state. Separated from the compute layer (trainer loop) to isolate side effects.

| Function | Purpose |
|---|---|
| `CheckpointManager.save(context, slots, strategy, scaler)` | Write manifest, model weights, optimizer/scheduler/scaler, and trainer state |
| `CheckpointManager.load(ckpt_dir)` | Read manifest checkpoints or legacy manifestless checkpoints |
| `save_checkpoint(state, ckpt_dir)` / `load_checkpoint(ckpt_dir)` | Legacy wrapper API retained for existing trainer call sites |
| `gather_fsdp_state_dict(model)` | Collect full state dict via all-rank FSDP collective |
| `export_model_artifact(model, metadata, mlflow_run, ...)` | Register policy/SFT artifact with MLflow |
| `find_best_checkpoint(strategy_config)` | Resolve best/latest symlink to checkpoint path |

New checkpoints contain `manifest.json` with `layout_version`, checkpoint kind, trainer state file, optional recipe/config snapshots, scaler path, and per-model records. Each model record declares role, weight format, relative path, trainable flag, and optional optimizer/scheduler files. Manifestless directories are treated as legacy checkpoints and are loaded best-effort from `trainer_state.json` and `scaler.pt`.

Strategies expose `checkpoint_capability`. DDP and FSDP opt in to manager-owned full-state checkpointing; FSDP also declares that save is an all-rank collective. The default strategy capability is unsupported, so DeepSpeed is intentionally fail-fast in the current Trainer/RLTrainer runtime. Its engine owns backward, optimizer step, ZeRO shards, and checkpoint semantics, so DeepSpeed ZeRO checkpoints must not be documented or restored as normal DDP/FSDP checkpoints until a separate engine-contract spec implements that path.

`Trainer` and `RLTrainer` call these via `BaseTrainer._checkpoint_state()` / `_load_checkpoint_state()` hooks — they no longer own save/resume/export logic directly.

### `_progress_log.py` — Progress Logging

Renamed from `_logging_helpers.py` (spec-training-restructure U2) to distinguish from `utils/logging.py` (system logging infrastructure). Contains the 6 free functions for trainer progress output:

`fmt_eta`, `log_step_progress`, `log_run_banner`, `dump_oom_summary`, `maybe_start_memory_history`, `maybe_dump_memory_snapshot`

`BaseTrainer` wraps these as bound methods so subclasses inherit them without re-importing.

### Inheritance diagram

```
BaseTrainer(ABC)
├── _base.py
│   ├── shim methods (delegating to _progress_log + _mlflow_logging)
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
  CLI commands and _torchrun_entry
  Load settings, callbacks, source plans, and launch Trainer/RLTrainer or serving paths
  ↓
Tier 2 (Composite)
  Factory        — component creation facade, singleton caching
  Trainer/RLTrainer — training loop execution
  (Factory and Trainer are both Tier 2 but do not create each other)
  ↓
Tier 1 (Atomic)
  Dataset, Transform, Optimizer, Scheduler, Loss, Head, Callback, Evaluator
```

Import direction flows top-to-bottom only. `import-linter` enforces this in CI. `Serving must not import training` is a hard contract — `BaseInferenceCallback` lives in `mdp/callbacks/base.py` (not `training/`) to satisfy this constraint.

## Key Design Decisions

**`_component_` pattern**: All pluggable components use `_component_: <alias or full.path>` in YAML. `ComponentResolver` dynamically imports and instantiates. No registry classes needed.

**Recipe / Config / Callbacks separation**: Recipe = what to train, Config = where to run it, Callbacks = side-channel observation/intervention. `--callbacks <yaml>` is the only callback injection path; `Recipe` has no `callbacks:` field.

**BaseTrainer inheritance over mixin**: `Trainer` and `RLTrainer` share a common lifecycle (init → mlflow start → epoch/step loop → checkpoint → mlflow end). A single abstract base class captures this cleanly; mixins would add MRO complexity.

**`_`-prefix = private to owning namespace**: `_base.py`, `_checkpoint.py`, `_features.py`, `_progress_log.py`, `_mlflow_logging.py`, `_schedulers.py`, `_common.py` are internal implementation modules. CLI entrypoints may import selected private helpers to bridge command-line YAML into runtime objects; serving and external consumer packages should not depend on training-private modules. Public trainer entrypoints remain `Trainer` and `RLTrainer`.

**`_features.py` is stateless**: Feature extractor dispatcher depends only on `(model, batch, layer_idx)` — no trainer state. This enables reuse from inference callbacks without instantiating a trainer.

**`losses/_ce_helpers.py` is stateless**: Sibling module to `_features.py`. While `_features.py` dispatches hidden state extraction, `_ce_helpers.py` provides the `compute_per_token_ce_chunked_from_hidden` free function for memory-efficient CE computation from hidden states. Both modules follow the same stateless free function principle (spec-training-restructure principle 4): no dependency on trainer or algorithm instance state, `chunk_size` passed as an explicit parameter so callers retain full control.

**Checkpoint I/O is a separate layer**: `_checkpoint.py` free functions are pure I/O. The trainer loop knows *when* to checkpoint; `_checkpoint.py` knows *how*. DDP/FSDP edge cases (collective state_dict gather, rank-0-only save) are isolated here.
