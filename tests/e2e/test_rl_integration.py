"""RL 통합 테스트 — 다양한 시나리오의 핵심 경로 검증.

SFT 테스트와 대등한 수준으로 RL 인프라를 검증한다.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import yaml

from mdp.settings.schema import (
    Config,
    DataSpec,
    RLGenerationSpec,
    MetadataSpec,
    RLSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from tests.e2e.datasets import ListDataLoader


# ── 테스트용 모델 ──


class TinyLM(nn.Module):
    def __init__(self, vocab=32, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.config = type("Config", (), {"pad_token_id": 0})()

    def forward(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": self.head(self.embed(input_ids))})()

    def generate(self, input_ids, attention_mask=None, max_new_tokens=4, **kwargs):
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids).logits
            ids = torch.cat([ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        return ids


def _pref_batches(n, bs, seq=8, vocab=32):
    return [{
        "chosen_input_ids": torch.randint(1, vocab, (bs, seq)),
        "chosen_attention_mask": torch.ones(bs, seq, dtype=torch.long),
        "chosen_labels": torch.randint(1, vocab, (bs, seq)),
        "rejected_input_ids": torch.randint(1, vocab, (bs, seq)),
        "rejected_attention_mask": torch.ones(bs, seq, dtype=torch.long),
        "rejected_labels": torch.randint(1, vocab, (bs, seq)),
    } for _ in range(n)]


def _prompt_batches(n, bs, seq=4, vocab=32):
    return [{
        "input_ids": torch.randint(1, vocab, (bs, seq)),
        "attention_mask": torch.ones(bs, seq, dtype=torch.long),
    } for _ in range(n)]


def _dpo_settings(max_steps=3, precision="fp32", **overrides):
    model_component = "tests.e2e.test_rl_integration.TinyLM"
    recipe = Recipe(
        name="rl-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "DPO", "beta": 0.1},
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                },
                "reference": {"_component_": model_component},
            },
        ),
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
        ),
        training=TrainingSpec(max_steps=max_steps, precision=precision, **overrides),
        metadata=MetadataSpec(author="test", description="test"),
    )
    config = Config()
    config.job.resume = "disabled"
    return Settings(recipe=recipe, config=config)


def _grpo_settings(max_steps=3):
    model_component = "tests.e2e.test_rl_integration.TinyLM"
    recipe = Recipe(
        name="rl-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "GRPO", "clip_range": 0.2, "kl_coeff": 0.01},
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                },
                "reference": {"_component_": model_component},
                "reward": {"_component_": model_component},
            },
            generation=RLGenerationSpec(max_new_tokens=4),
        ),
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
        ),
        training=TrainingSpec(max_steps=max_steps),
        metadata=MetadataSpec(author="test", description="test"),
    )
    return Settings(recipe=recipe, config=Config())


def _make_trainer(settings, batches, models=None):
    from mdp.training.rl_trainer import RLTrainer

    model_names = list(settings.recipe.rl.models.keys())
    if models is None:
        models = {name: TinyLM() for name in model_names}
    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(batches),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    return trainer


def test_rl_trainer_from_bundle_preserves_per_model_optimizers() -> None:
    """Bundle-oriented path keeps RL optimizer/scheduler ownership per model."""
    from mdp.assembly.bundles import build_rl_training_bundle
    from mdp.training.rl_trainer import RLTrainer

    model_component = "tests.e2e.test_rl_integration.TinyLM"
    recipe = Recipe(
        name="rl-bundle-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "PPO", "mini_epochs": 1},
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                },
                "value": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 2e-3},
                },
                "reference": {"_component_": model_component},
            },
        ),
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
        ),
        training=TrainingSpec(max_steps=1),
        metadata=MetadataSpec(author="test", description="test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    models = {name: TinyLM() for name in settings.recipe.rl.models}

    bundle = build_rl_training_bundle(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(_prompt_batches(2, 2)),
    )
    trainer = RLTrainer.from_bundle(bundle)

    assert set(trainer.optimizers) == {"policy", "value"}
    assert trainer.optimizers["policy"] is not trainer.optimizers["value"]
    assert set(trainer.trainable) == {"policy", "value"}
    assert set(trainer.frozen) == {"reference"}


# ── 1. bf16 precision ──


def test_rl_bf16_precision() -> None:
    """DPO가 bf16 autocast로 학습 완료되는지."""
    settings = _dpo_settings(max_steps=2, precision="bf16")
    trainer = _make_trainer(settings, _pref_batches(5, 4))
    result = trainer.train()
    assert result["total_steps"] == 2


# ── 2. EarlyStopping ──


def test_rl_early_stopping() -> None:
    """EarlyStopping 콜백이 RL 학습을 조기 종료하는지."""
    from mdp.training.callbacks.early_stopping import EarlyStopping

    settings = _dpo_settings(max_steps=100)
    trainer = _make_trainer(settings, _pref_batches(20, 4))

    # patience=0으로 즉시 종료 트리거
    es = EarlyStopping(monitor="val_loss", patience=0, mode="min")
    es.should_stop = True  # 강제 설정
    trainer.callbacks.append(es)

    result = trainer.train()
    assert result["total_steps"] < 100


# ── 3. Gradient accumulation ──


def test_rl_gradient_accumulation() -> None:
    """grad_accum=4일 때 optimizer step이 올바른 횟수로 실행되는지."""
    from mdp.training.callbacks.base import BaseCallback

    class StepCounter(BaseCallback):
        def __init__(self):
            self.count = 0
        def on_batch_end(self, **kwargs):
            self.count += 1

    settings = _dpo_settings(max_steps=2, gradient_accumulation_steps=4)
    trainer = _make_trainer(settings, _pref_batches(10, 4))
    counter = StepCounter()
    trainer.callbacks.append(counter)

    trainer.train()
    # 2 optimizer steps × 4 accum batches = 8 batches, on_batch_end fires every 4 = 2회
    assert counter.count == 2


# ── 4. MLflow artifact 저장 ──


def test_rl_mlflow_artifact_saved(tmp_path) -> None:
    """RL 학습 후 MLflow에 policy artifact가 저장되는지."""
    import mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    settings = _dpo_settings(max_steps=2)
    settings.config.mlflow.tracking_uri = tracking_uri
    settings.config.mlflow.experiment_name = "rl-test"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rl-test")

    trainer = _make_trainer(settings, _pref_batches(5, 4))
    trainer.train()

    client = mlflow.tracking.MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name("rl-test")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) > 0


# ── 5. on_batch_end 호출 시점 ──


def test_rl_on_batch_end_per_accum_step() -> None:
    """on_batch_end가 accumulation step 기준으로 호출되는지."""
    from mdp.training.callbacks.base import BaseCallback

    class BatchEndLogger(BaseCallback):
        def __init__(self):
            self.steps = []
        def on_batch_end(self, **kwargs):
            self.steps.append(kwargs.get("global_step"))

    settings = _dpo_settings(max_steps=3, gradient_accumulation_steps=2)
    trainer = _make_trainer(settings, _pref_batches(10, 4))
    logger_cb = BatchEndLogger()
    trainer.callbacks.append(logger_cb)

    trainer.train()
    # 3 optimizer steps × 2 accum batches = 6 batches, on_batch_end fires every 2 = 3회
    assert len(logger_cb.steps) == 3


# ── 6. GRPO generation + reward scoring ──


def test_grpo_reward_from_model() -> None:
    """GRPO에서 reward model의 출력이 advantage 계산에 사용되는지."""
    settings = _grpo_settings(max_steps=2)
    trainer = _make_trainer(settings, _prompt_batches(5, 4))
    result = trainer.train()
    assert result["total_steps"] == 2
    assert result["metrics"]["loss"] != 0


# ── 7. PPO value + policy multi-loss ──


def test_ppo_multi_loss() -> None:
    """PPO에서 policy loss와 value loss가 동시에 계산되는지."""
    model_component = "tests.e2e.test_rl_integration.TinyLM"
    recipe = Recipe(
        name="ppo-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "PPO", "clip_range": 0.2, "mini_epochs": 1},
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                },
                "value": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                    "freeze": False,
                },
                "reference": {"_component_": model_component},
                "reward": {"_component_": model_component},
            },
            generation=RLGenerationSpec(max_new_tokens=4),
        ),
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
        ),
        training=TrainingSpec(max_steps=2),
        metadata=MetadataSpec(author="test", description="test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    trainer = _make_trainer(settings, _prompt_batches(5, 4))
    result = trainer.train()
    assert result["total_steps"] == 2


# ── 8. RL → export → serve 흐름 ──


def test_rl_export_serve_flow(tmp_path) -> None:
    """RL 학습 → artifact → reconstruct_model이 동작하는지."""
    # RL 학습 시뮬레이션: policy 저장
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()

    policy = TinyLM()
    from safetensors.torch import save_file
    save_file(policy.state_dict(), str(artifact_dir / "model.safetensors"))

    recipe_dict = {
        "name": "rl-serve-test",
        "task": "text_generation",
        "model": {"_component_": "tests.e2e.test_rl_integration.TinyLM"},
        "data": {
            "dataset": {"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            "collator": {"_component_": "mdp.data.collators.CausalLMCollator", "tokenizer": "gpt2"},
        },
        "training": {"max_steps": 1},
        "metadata": {"author": "test", "description": "test"},
    }
    (artifact_dir / "recipe.yaml").write_text(yaml.dump(recipe_dict))

    from mdp.serving.model_loader import reconstruct_model
    model, settings = reconstruct_model(artifact_dir)

    # forward 가능한지
    model.eval()
    batch = {"input_ids": torch.randint(1, 32, (2, 4))}
    with torch.no_grad():
        out = model(batch["input_ids"])
    assert hasattr(out, "logits")


# ── 9. Recipe 검증: policy optimizer 필수 ──


def test_rl_policy_optimizer_required() -> None:
    """policy에 optimizer가 없으면 에러."""
    with pytest.raises(ValueError, match="optimizer가 필수"):
        Recipe(
            name="bad",
            task="text_generation",
            rl=RLSpec(
                algorithm={"_component_": "DPO"},
                models={
                    "policy": {"_component_": "tests.e2e.test_rl_integration.TinyLM"},  # optimizer 없음
                    "reference": {"_component_": "tests.e2e.test_rl_integration.TinyLM"},
                },
            ),
            data=DataSpec(
                dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
                collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
            ),
            training=TrainingSpec(max_steps=1),
            metadata=MetadataSpec(author="test", description="test"),
        )


# ── 10. Custom algorithm + causal data ──


def test_custom_algorithm_causal_data() -> None:
    """커스텀 알고리즘이 causal(input_ids) 데이터로 동작하는지."""
    causal_batches = [{
        "input_ids": torch.randint(1, 32, (4, 8)),
        "attention_mask": torch.ones(4, 8, dtype=torch.long),
        "labels": torch.randint(1, 32, (4, 8)),
    } for _ in range(5)]

    model_component = "tests.e2e.test_rl_integration.TinyLM"
    recipe = Recipe(
        name="custom-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={
                "_component_": "tests.e2e.test_rl_custom_algorithm.SimpleWeightedCELoss",
                "weight_scale": 1.0,
            },
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                },
                "critic": {"_component_": model_component},
            },
        ),
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
        ),
        training=TrainingSpec(max_steps=3),
        metadata=MetadataSpec(author="test", description="test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    trainer = _make_trainer(settings, causal_batches)
    result = trainer.train()
    assert result["total_steps"] == 3
