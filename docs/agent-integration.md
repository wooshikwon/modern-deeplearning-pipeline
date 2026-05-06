# Agent Integration

AI agents should treat MDP as a JSON-speaking CLI with explicit discovery,
configuration, and recovery contracts.

## Scope

This document owns agent workflow only:

- how to discover tasks/models
- how to scaffold a run
- how to choose the next action from structured results
- where to find the canonical references

Command flags live in [CLI Reference](cli-reference.md). JSON schemas, MLflow,
logging, graceful shutdown, and recovery policy live in
[Observability](observability.md). YAML structure lives in
[Configuration Guide](configuration.md). Runtime compatibility rules live in
[Runtime Contracts](runtime-contracts.md).

## Discovery Flow

```bash
mdp list tasks --format json
mdp list models --task text_generation --format json
mdp init my_project --task text_generation --model llama3-8b
mdp estimate -r my_project/recipes/example.yaml --format json
mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml --format json
mdp inference --run-id <id> --data test.jsonl --format json
```

After `mdp init`, fill only the generated `???` fields before estimating or
training.

## Autonomous Loop

A typical agent loop:

1. Run `mdp estimate` and choose a supported strategy.
2. Run `mdp train` or `mdp rl-train` with `--format json`.
3. Inspect `status`, `stopped_reason`, and `checkpoints_saved`.
4. If no checkpoint was written, fix callback/monitor configuration before
   continuing.
5. Evaluate with `mdp inference`.
6. Use `--override` for controlled sweeps instead of rewriting YAML for every
   small variation.

Example sweep commands are in [CLI Reference](cli-reference.md#global-options).
Recovery decisions are in [Observability](observability.md#error-recovery).

## Responsibility Boundary

| Area | User/agent owns | MDP owns |
|---|---|---|
| Data | Collection, preprocessing, storage format | Loading, tokenization, augmentation hooks |
| Model | Custom `BaseModel` when needed | Pretrained loading, head replacement, adapter application |
| Infrastructure | GPU provisioning, SSH/cloud/K8s orchestration | Local process launch, internal torchrun, runtime validation |
| Training | Recipe/Config authoring and sweep decisions | Loop, AMP, checkpoints, MLflow, callbacks |
| Serving | Load balancer, autoscale, DNS | FastAPI app, batching/streaming hooks |
| Inference | Business post-processing | Batch inference, generation, drift checks |

## Entrypoint

Root [AGENT.md](../AGENT.md) is the short start page for agents. Keep detailed
contracts in the canonical docs linked above.
