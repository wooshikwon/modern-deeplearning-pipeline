# MDP Agent Entrypoint

This file is the short operating guide for AI agents using or modifying MDP.
Detailed references live in `docs/`.

## First Read

- [Getting Started](docs/getting-started.md): install, scaffold, train, infer, serve.
- [CLI Reference](docs/cli-reference.md): commands, flags, model source rules, overrides.
- [YAML Schema](docs/configuration.md): Recipe, Config, Callbacks YAML.
- [Runtime Contracts](docs/runtime-contracts.md): validation, distributed, checkpoint, source precedence.
- [Training Guide](docs/training.md): SFT, RL, checkpoints, MLflow, graceful shutdown.
- [Inference & Serving](docs/inference-and-serving.md): batch inference, generation, REST serving.
- [Extending MDP](docs/extending.md): custom models, callbacks, datasets, samplers, strategies, RL algorithms.
- [Observability](docs/observability.md): JSON output, logging, MLflow, error recovery.
- [Architecture](docs/architecture.md): internal module boundaries and design decisions.

## Command Map

| Command | Purpose |
|---|---|
| `mdp init <name> --task <task> --model <model>` | Scaffold a project and create Recipe/Config templates. |
| `mdp train -r recipe.yaml -c config.yaml` | Run SFT training. |
| `mdp rl-train -r rl-recipe.yaml -c config.yaml` | Run RL alignment training: DPO, GRPO, PPO, or injected algorithm. |
| `mdp inference` | Run batch inference/evaluation. |
| `mdp generate` | Run autoregressive text generation from JSONL prompts. |
| `mdp estimate -r recipe.yaml` | Estimate GPU memory and suggest a supported strategy. |
| `mdp export --run-id <id> --output <dir>` | Export/package a trained artifact. |
| `mdp serve --run-id <id>` | Serve a trained artifact with REST API. |
| `mdp list models\|tasks\|callbacks\|strategies` | Discover catalogs and extension aliases. |

All commands support `--format json`. `train`, `rl-train`, `inference`, and
`generate` support `--override` with either repeated `KEY=VALUE` arguments or a
JSON object.

## Agent Discovery Flow

```bash
mdp list tasks --format json
mdp list models --task text_generation --format json
mdp init my_project --task text_generation --model llama3-8b
# Fill only the ??? fields in the generated Recipe.
mdp estimate -r my_project/recipes/example.yaml --format json
mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml --format json
mdp inference --run-id <id> --data test.jsonl --format json
```

## Hard Rules

- Do not wrap `mdp train` or `mdp rl-train` in external `torchrun`. MDP starts
  internal torchrun when `compute.distributed.strategy` is set and multiple GPUs
  are requested.
- `compute.distributed.strategy` currently supports `ddp`, `fsdp`, and `auto`.
  `deepspeed*` aliases are listed for forward compatibility but fail fast until
  a separate DeepSpeed engine-contract implementation exists.
- Recipe and Config are separate. Recipe defines the experiment; Config defines
  infrastructure/runtime.
- Recipe has no `callbacks:` field. Pass callbacks with `--callbacks callbacks.yaml`.
  EarlyStopping and EMA are first-class `training.*` Recipe fields because they
  affect training semantics.
- Model source flags are mutually exclusive. Use exactly one of `--run-id`,
  `--model-dir`, or `--pretrained`. `--pretrained` is only for `inference` and
  `generate`; `serve` and `export` require an artifact/checkpoint source.
- `device_map` is inference/serving-only. Training must use DDP/FSDP through
  `compute.distributed.strategy`.
- External consumers should import `Trainer` or `RLTrainer`, not `BaseTrainer`.
  `BaseTrainer` is an abstract lifecycle base class.

## Canonical Contracts

- Checkpoint/source/distributed contracts: [Runtime Contracts](docs/runtime-contracts.md).
- JSON output, MLflow, logging, graceful shutdown, and recovery:
  [Observability](docs/observability.md).
- Command flags and override syntax: [CLI Reference](docs/cli-reference.md).
- YAML schemas: [Configuration Guide](docs/configuration.md).

## Common Pitfalls

- `compute.gpus: auto` alone does not enable distributed training. Set
  `compute.distributed.strategy`.
- `ModelCheckpoint.strict: true` catches monitor-name typos at first validation.
- Long-running `inference` and `generate` do not currently install the training
  signal handler. A timeout can leave partial output.
- `--save-output` is needed when using inference callbacks but still wanting the
  default prediction file.

## Maintainer Notes

Keep this file small. When adding detailed behavior, place it in the relevant
`docs/` file and link it from here. This file should answer "where do I start?"
and "what must an agent not get wrong?", not duplicate the full public docs.
