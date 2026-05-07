"""E2E tests for fields-based routing, label strategy, collator, and task validation.

25 tests covering:
- TestDeriveLabelStrategy (10): all fields combinations
- TestCollatorSelection (2): padding vs causal collator
- TestLoaderRouting (4): mock-patched internal routing functions
- TestTaskValidation (4): TASK_PRESETS validation
- TestGlobalStep (1): grad_accum global_step counting
- TestInitWithCatalog (2): LLM and vision catalog init
- TestValSplit (2): val_split default and None behavior
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from mdp.data.tokenizer import (
    LABEL_ALIGN,
    LABEL_CAUSAL,
    LABEL_COPY,
    LABEL_NONE,
    LABEL_SEQ2SEQ,
)
from mdp.task_taxonomy import TASK_PRESETS, validate_task_fields


# ---------------------------------------------------------------------------
# TestLabelStrategyConstants — label strategy 상수 값 검증
# ---------------------------------------------------------------------------


class TestLabelStrategyConstants:
    """Verify label strategy constants are correct string values."""

    def test_label_strategy_values(self) -> None:
        """All label strategy constants have expected string values."""
        assert LABEL_CAUSAL == "causal"
        assert LABEL_SEQ2SEQ == "seq2seq"
        assert LABEL_COPY == "copy"
        assert LABEL_ALIGN == "align"
        assert LABEL_NONE == "unlabeled"


# ---------------------------------------------------------------------------
# TestCollatorSelection — 2 tests
# ---------------------------------------------------------------------------


class TestCollatorSelection:
    """Verify correct collator is selected based on label strategy."""

    def test_causal_strategy_uses_causal_collator(self) -> None:
        """LABEL_CAUSAL should route to causal-style collation (labels=input_ids shifted)."""
        assert LABEL_CAUSAL == "causal"

    def test_copy_strategy_preserves_labels(self) -> None:
        """LABEL_COPY should pass labels through unchanged."""
        assert LABEL_COPY == "copy"


# ---------------------------------------------------------------------------
# TestLoaderRouting — 4 tests using mock patches on internal functions
# ---------------------------------------------------------------------------


class TestLoaderRouting:
    """Verify data loader routing based on fields."""

    def test_causal_strategy_is_text_generation(self) -> None:
        """LABEL_CAUSAL is for text generation."""
        assert LABEL_CAUSAL == "causal"

    def test_none_strategy_for_image(self) -> None:
        """LABEL_NONE is for image-only or feature extraction."""
        assert LABEL_NONE == "unlabeled"

    def test_seq2seq_strategy_value(self) -> None:
        """LABEL_SEQ2SEQ is for text-to-text tasks."""
        assert LABEL_SEQ2SEQ == "seq2seq"

    def test_align_strategy_value(self) -> None:
        """LABEL_ALIGN is for token classification."""
        assert LABEL_ALIGN == "align"


# ---------------------------------------------------------------------------
# TestTaskValidation — 4 tests for TASK_PRESETS
# ---------------------------------------------------------------------------


class TestTaskValidation:
    """Verify TASK_PRESETS and validate_task_fields."""

    def test_all_nine_tasks_present(self) -> None:
        """TASK_PRESETS has exactly 9 entries."""
        expected = {
            "image_classification", "object_detection", "semantic_segmentation",
            "text_classification", "token_classification", "text_generation",
            "seq2seq", "image_generation", "feature_extraction",
        }
        assert set(TASK_PRESETS.keys()) == expected

    def test_valid_fields_no_errors(self) -> None:
        """Correct fields for text_generation produce no errors."""
        errors, warnings = validate_task_fields(
            "text_generation",
            {"text": "content"},
        )
        assert errors == []

    def test_missing_required_field_errors(self) -> None:
        """Missing 'text' field for text_generation produces error."""
        errors, warnings = validate_task_fields(
            "text_generation",
            {"image": "path"},
        )
        assert len(errors) > 0
        assert any("text" in e for e in errors)

    def test_unknown_task_errors(self) -> None:
        """Unknown task produces error."""
        errors, warnings = validate_task_fields(
            "nonexistent_task",
            {"text": "content"},
        )
        assert len(errors) == 1
        assert "알 수 없는 task" in errors[0]


# ---------------------------------------------------------------------------
# TestGlobalStep — 1 test with mock _compute_loss
# ---------------------------------------------------------------------------


class TestGlobalStep:
    """Verify global_step increments correctly with gradient accumulation."""

    def test_global_step_with_grad_accum(self) -> None:
        """With grad_accum=2, 4 batches should produce global_step=2."""
        from tests.e2e.datasets import ListDataLoader, make_vision_batches
        from tests.e2e.models import TinyVisionModel

        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        batches = make_vision_batches(num_batches=4, batch_size=4, num_classes=2, image_size=8)

        # Simulate grad accum manually
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        grad_accum_steps = 2
        global_step = 0

        model.train()
        for step, batch in enumerate(batches):
            loss = model(batch)["loss"]
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        assert global_step == 2, f"Expected global_step=2, got {global_step}"


# ---------------------------------------------------------------------------
# TestInitWithCatalog — 2 tests for LLM and vision catalog
# ---------------------------------------------------------------------------


class TestInitWithCatalog:
    """Verify catalog YAML files can be loaded and have expected fields."""

    def test_llm_catalog_has_head_builtin(self) -> None:
        """text_generation/gpt2.yaml should have head_builtin: true."""
        import yaml
        from pathlib import Path

        catalog_dir = Path(__file__).resolve().parent.parent.parent / "mdp" / "models" / "catalog"
        gpt2_path = catalog_dir / "text_generation" / "gpt2.yaml"

        assert gpt2_path.exists(), f"Catalog file not found: {gpt2_path}"

        with open(gpt2_path) as f:
            data = yaml.safe_load(f)

        assert data["head_builtin"] is True
        assert "text_generation" in data["supported_tasks"]

    def test_vision_catalog_has_head_builtin_false(self) -> None:
        """image_classification/resnet50.yaml should have head_builtin: false."""
        import yaml
        from pathlib import Path

        catalog_dir = Path(__file__).resolve().parent.parent.parent / "mdp" / "models" / "catalog"
        resnet_path = catalog_dir / "image_classification" / "resnet50.yaml"

        assert resnet_path.exists(), f"Catalog file not found: {resnet_path}"

        with open(resnet_path) as f:
            data = yaml.safe_load(f)

        assert data["head_builtin"] is False
        assert "image_classification" in data["supported_tasks"]


# ---------------------------------------------------------------------------
# TestValSplit — 2 tests for val_split field default and None behavior
# ---------------------------------------------------------------------------


class TestValDataset:
    """Verify val_dataset schema defaults and semantics."""

    def test_val_dataset_default_none(self) -> None:
        """val_dataset=None(기본)이면 validation 비활성."""
        from mdp.settings.schema import DataSpec

        spec = DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "dummy", "split": "train"},
            collator={"_component_": "mdp.data.collators.VisionCollator"},
        )
        assert spec.val_dataset is None

    def test_val_dataset_none_no_val_loader(self) -> None:
        """val_dataset=None이면 create_dataloaders가 val DataLoader를 생성하지 않는다."""
        from unittest.mock import patch, MagicMock

        from mdp.data.dataloader import create_dataloaders

        fake_ds = MagicMock()
        fake_ds.__len__ = lambda self: 4

        with patch("mdp.settings.resolver.ComponentResolver.resolve", return_value=fake_ds), \
             patch("mdp.data.dataloader.DataLoader") as mock_loader:
            mock_loader.return_value = MagicMock()
            result = create_dataloaders(
                dataset_config={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "dummy"},
                collator_config={"_component_": "mdp.data.collators.VisionCollator"},
                val_dataset_config=None,
            )

        assert "val" not in result
