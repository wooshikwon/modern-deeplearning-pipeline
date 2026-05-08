# Contract Test Inventory

This directory is the landing zone for tests that protect MDP's stable runtime
contracts. U1 is inventory-only: it does not change test behavior, pytest
markers, assertions, or production code.

Current collection baseline after U8:

- Command: `uv run pytest --collect-only -q`
- Result: `1451 tests collected`

## Ownership Map

| Area | Current files | U2-U8 owner | Classification | Migration note |
|------|---------------|-------------|----------------|----------------|
| Settings loading, overrides, artifact snapshots | `tests/unit/test_settings_materializer.py`, `tests/unit/test_override.py`, `tests/unit/test_override_integration.py`, `tests/unit/test_recipe_fixtures.py`, `tests/e2e/test_settings.py` | `tests/contract/test_run_plan.py` | keep / migrate | Keep schema and pure fixture validation in unit; move stable `RunPlan` fields and facade parity to contract. |
| Launch and distributed intent | `tests/unit/test_launch_strategy_contract.py`, `tests/e2e/test_distributed_cpu.py`, `tests/e2e/test_distributed_rl.py`, `tests/e2e/test_gpu_distributed_sft.py`, `tests/e2e/test_gpu_distributed_rl.py` | `tests/contract/test_launch_plan.py`, `tests/contract/test_execution_engine.py` | migrate / quarantine | Replace private launcher monkeypatch expectations with `LaunchPlan`; keep CPU/GPU torchrun execution under existing markers. |
| Assembly graph shape | `tests/unit/test_load_pretrained_routing.py`, `tests/unit/test_materializer_semantic.py`, `tests/e2e/test_materializer_integration.py`, `tests/unit/test_recipe_fixtures.py` | `tests/contract/test_assembly_plan.py` | keep / migrate | Plan node shape and owned config belong to contract; pure route helpers and family routing can remain unit-owned. |
| Materialization and AssemblyMaterializer facade parity | `tests/e2e/test_materializer_integration.py`, `tests/e2e/test_materializer_e2e.py`, `tests/unit/test_resolver_positional_args.py`, `tests/unit/test_pretrained_auto_class.py` | `tests/contract/test_materializer.py` | migrate / keep | Contract should assert `AssemblyPlan -> bundle` and `AssemblyMaterializer(AssemblyPlan)` parity; low-level resolver behavior stays unit-owned. |
| Execution engine and training dispatch | `tests/e2e/test_trainer_e2e.py`, `tests/e2e/test_rl_integration.py`, `tests/e2e/test_rl_dpo.py`, `tests/unit/test_base_trainer.py`, `tests/unit/test_rl_trainer_*` | `tests/contract/test_execution_engine.py` | keep | Engine result normalization and wrapper parity moved to contract; algorithm math and trainer internals stay unit/e2e-owned. |
| CLI command surface and JSON error contract | `tests/e2e/test_cli.py`, `tests/e2e/test_cli_error_contract.py`, `tests/unit/test_generate_cli.py`, `tests/unit/test_cli_model_source.py`, `tests/unit/test_cli_setup_logging.py` | `tests/e2e/test_cli_train_contract.py`, existing CLI e2e files | keep / migrate | Actual command entry, `--format`, stdout JSON, and stderr separation are CLI-e2e contract. |
| Artifact and checkpoint contract | `tests/unit/test_checkpoint.py`, `tests/unit/test_checkpoint_manifest.py`, `tests/unit/test_checkpoint_monitor.py`, `tests/e2e/test_checkpoint_recipe.py`, `tests/e2e/test_model_loader.py`, `tests/e2e/test_mlflow_artifacts.py`, `tests/e2e/test_resume.py` | `tests/contract/test_artifact_checkpoint.py` | keep / migrate | Manifest serialization stays unit; source/artifact load plan and checkpoint output contract move to contract. |
| Runtime import boundaries | `tests/unit/test_fresh_imports.py`, import-linter contracts in `pyproject.toml` | `tests/contract/test_runtime_imports.py` | keep / migrate | Fresh import smoke remains quick regression; architectural import boundary can be represented as contract. |
| Data, fields, and validation utilities | `tests/e2e/test_data_*`, `tests/e2e/test_fields_routing.py`, `tests/unit/test_dataloader_no_sampler_regression.py`, `tests/unit/test_dataset_lengthed.py`, sampler tests | existing unit/e2e | keep | These protect schema, routing, and sampler behavior rather than runtime contract ownership. |
| Model forward, inference, generation, serving | `tests/e2e/test_forward_adapter.py`, `tests/e2e/test_inference*.py`, `tests/e2e/test_generate.py`, `tests/e2e/test_serve_endpoint.py`, model unit tests | existing unit/e2e | keep | Keep as component and command behavior tests unless later engine contracts consume a shared result boundary. |
| GPU, fixture, distributed, and memory execution | `tests/e2e/test_gpu_*.py`, marked distributed tests, `tests/unit/test_gpu_fixture_contract.py` | marker/cloud boundary in U8 | quarantine / keep | Preserve existing marker semantics; do not pull fixture-tier backlog into this spec. |

## Classification Rules

| Classification | Rule |
|----------------|------|
| keep | Test protects a stable contract or a pure utility and can survive runtime refactors. |
| migrate | Test protects meaningful risk but is coupled to private seams or direct internal assembly. |
| delete | Test becomes redundant only after an equivalent contract test exists and passes. |
| quarantine | Test requires GPU, distributed hardware, fixture artifacts, memory profiles, or slow real-model loading. |

## Marker Semantics

Marker behavior is unchanged in this realignment.

| Marker | Meaning | Current skip owner |
|--------|---------|--------------------|
| `slow` | Real model loading or long integration path; deselect with `-m 'not slow'`. | pytest marker registration |
| `gpu` | Requires at least one CUDA GPU. | `tests/conftest.py` |
| `distributed` | Requires at least two CUDA GPUs. | `tests/conftest.py` |
| `distributed_cpu` | CPU gloo torchrun path; opt in with `MDP_RUN_DISTRIBUTED_CPU=1`. | `tests/conftest.py` |
| `fixtures` | Requires prepared `MDP_TEST_FIXTURES`. | `tests/conftest.py` |
| `memory` | Requires `MDP_MEMORY_BUDGET_PROFILE`. | `tests/conftest.py` |

## Churn Ledger

### `tests/unit/test_load_pretrained_routing.py`

| Item | Protects | New owner | Classification | Deletion condition |
|------|----------|-----------|----------------|--------------------|
| `TestModelLoadRouteDecision` | Route selection for component, pretrained, QLoRA, and missing source. | unit | keep | Delete only if model source routing becomes fully represented by a public `ModelSourcePlan` contract and the helper is removed. |
| `TestAssemblyPlanModelRouteShape` | Assembly plan preserves model route inputs without materializing components. | `tests/contract/test_assembly_plan.py` | migrate | Delete after U3 has equivalent model node, adapter ownership, and distributed intent assertions. |
| `TestCustomBaseModelWithPretrained` | Custom `BaseModel` receives `pretrained` and extra kwargs through materialization. | `tests/contract/test_materializer.py` | migrate | Delete after U3 proves materializer/facade parity for custom pretrained construction. |
| `TestHFClassWithPretrained` | HF-like class uses `from_pretrained` with URI normalization. | `tests/contract/test_materializer.py` or unit | migrate | Delete only after materializer contract covers URI normalization; otherwise keep as unit route regression. |
| `TestComponentOnlyNoPretrained` | Component constructor path without pretrained. | `tests/contract/test_materializer.py` | delete | Consumed in U5; `test_policy_model_materialization_matches_materializer_facade_shape` covers component constructor materialization through public materializer/facade path. |
| `TestPretrainedOnlyNoComponent` | Pretrained-only path delegates to `PretrainedResolver`. | `tests/contract/test_materializer.py` | delete | Consumed in U5; `test_pretrained_only_model_node_uses_pretrained_resolver_boundary` covers pretrained-only materialization through the contract owner. |
| `TestBaseModelValidation` | Custom classes must satisfy `BaseModel`; HF `from_pretrained` path is exempt. | unit | keep | Delete only if validation moves to schema/contract layer with equivalent error assertions. |

### `tests/unit/test_materializer_semantic.py`

| Item | Protects | New owner | Classification | Deletion condition |
|------|----------|-----------|----------------|--------------------|
| `TestResolveSemantic` | Semantic adapter/head keys translate to raw module names and reject conflicts. | unit | keep | Delete only if semantic translation leaves `AssemblyMaterializer` and gains a public pure helper contract. |
| `TestAssemblyPlanSemanticBoundary` | Assembly plan preserves semantic config unresolved for materialization. | `tests/contract/test_assembly_plan.py` | delete | Consumed in U5; `test_sft_recipe_builds_policy_only_component_graph` and `test_qlora_route_inputs_remain_owned_by_model_node` cover semantic key preservation in plan nodes. |
| `TestResolveSemanticFromConfig` | QLoRA semantic resolution from HF config, conflict handling, and non-HF fail-fast. | unit | keep | Delete only if QLoRA route resolution is exposed through a stable materializer contract with equivalent assertions. |

### `tests/e2e/test_pipeline_e2e.py`

| Item | Protects | New owner | Classification | Deletion condition |
|------|----------|-----------|----------------|--------------------|
| `test_yaml_to_training_e2e` | YAML settings, materializer, trainer, and CPU training result can complete together. | `tests/contract/test_execution_engine.py` plus CLI train e2e | delete | Consumed before U5; engine and CLI train contracts assert equivalent runtime/result fields. |
| `test_init_generates_parseable_yaml` | `mdp init` fallback template is parseable and component-unified. | `tests/e2e/test_cli.py` | delete | Consumed in U5; fallback parseability and `_component_` schema checks moved into `test_init_project`. |
| `test_catalog_init_recipes_are_parseable_for_all_supported_tasks` | Catalog-supported tasks produce valid `Recipe` objects. | `tests/e2e/test_cli.py` | delete | Consumed in U5; catalog parseability lives in the parameterized CLI init matrix. |
| `test_catalog_init_recipes_use_task_specific_data_components` | Catalog init uses task-specific dataset and collator components. | `tests/e2e/test_cli.py` | delete | Consumed in U5; task-specific dataset/collator assertions were folded into the CLI init matrix. |
| `test_train_json_output_schema` | Trainer result can populate `TrainResult` JSON schema fields. | `tests/contract/test_execution_engine.py` and `tests/e2e/test_cli_train_contract.py` | delete | Consumed before U5; engine/CLI contracts check result shape, including steps, stopped reason, output, and checkpoint fields. |

## Out Of Scope For This Spec

The following coverage remains owned by the fixture CLI e2e follow-up backlog:

- PPO or GRPO fixture-tier e2e with `prompt_tiny`
- `bert-tiny` text evaluation e2e
- DDP checkpoint/resume two-stage validation
- FSDP `needs_hidden_states=True` fixture boundary
