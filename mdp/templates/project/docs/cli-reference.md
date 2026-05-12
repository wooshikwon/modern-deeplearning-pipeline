# CLI Reference

This document collects command-line behavior that agents need to call MDP
programmatically.

## Global Options

`--format text|json` is accepted globally and on subcommands. JSON mode writes
structured results to stdout and leaves logs/errors on stderr.

`train`, `rl-train`, `inference`, and `generate` accept `--override`.

```bash
mdp train -r recipe.yaml -c config.yaml \
  --override training.epochs=0.1 \
  --override data.dataloader.batch_size=8

mdp train -r recipe.yaml -c config.yaml \
  --override '{"training.epochs": 0.1, "data.dataloader.batch_size": 8}'
```

Override values are parsed as `null`/`none`, booleans, integers, floats, JSON
objects/arrays, then strings.

## Commands

| Command | Required inputs | Notes |
|---|---|---|
| `mdp init` | `name`, optionally `--task`, `--model` | Creates project folders, Recipe, Config, and placeholders. |
| `mdp train` | `-r/--recipe`, `-c/--config` | SFT training. `--callbacks` injects a callback YAML file. |
| `mdp rl-train` | `-r/--recipe`, `-c/--config` | RL training for DPO/GRPO/PPO or custom algorithms. |
| `mdp inference` | one model source plus `--data` | Batch inference/evaluation. |
| `mdp generate` | one model source plus `--prompts` | JSONL prompt generation. |
| `mdp estimate` | `-r/--recipe` | Memory estimate and strategy suggestion. |
| `mdp export` | `--run-id` or `--checkpoint` | Produces a serving package. |
| `mdp serve` | `--run-id` or `--model-dir` | Starts REST server. |
| `mdp list` | `models`, `tasks`, `callbacks`, or `strategies` | Catalog discovery. |

## Model Sources

`inference` and `generate` accept exactly one of:

- `--run-id <id>`: MLflow run artifact.
- `--model-dir <path>`: local export or checkpoint directory.
- `--pretrained <uri>`: direct open-source model load.

`serve` accepts `--run-id` or `--model-dir`. `export` accepts `--run-id` or
`--checkpoint`. `--pretrained` is not valid for `serve` or `export`.

Supported pretrained URI prefixes include `hf://`, `timm://`, `ultralytics://`,
and `local://`.

## Inference Flags

```text
--run-id <id>
--model-dir <path>
--pretrained <uri>
--tokenizer <name>
--data <path>
--fields role=col ...
--metrics Metric ...
--callbacks <yaml>
--save-output
--output-format parquet|csv|jsonl
--output-dir <path>
--device-map auto|balanced|sequential
--dtype float32|float16|bfloat16
--trust-remote-code
--attn-impl flash_attention_2|sdpa|eager
--batch-size N
--max-length N
```

## Generate Flags

```text
--run-id <id>
--model-dir <path>
--pretrained <uri>
--tokenizer <name>
--prompts <jsonl>
--prompt-field <name>
-o, --output <path>
--max-new-tokens N
--temperature F
--top-p F
--top-k N
--do-sample
--num-samples N
--batch-size N
--callbacks <yaml>
--device-map auto|balanced|sequential
--dtype float32|float16|bfloat16
--trust-remote-code
--attn-impl flash_attention_2|sdpa|eager
```

Generation kwargs precedence is:

```text
MDP defaults < Recipe generation < explicit CLI/serving args
```

## Distributed Launch

Do not run `torchrun ... mdp train ...` or `torchrun ... mdp rl-train ...`.
When distributed training is configured, MDP launches its own internal torchrun
entrypoint.

Distributed training requires `compute.distributed.strategy`. `compute.gpus:
auto` alone does not start distributed execution.
