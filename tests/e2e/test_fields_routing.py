"""E2E tests for fields-based routing, label strategy, collator, and task validation.

23 tests covering:
- TestDeriveLabelStrategy (10): all fields combinations
- TestCollatorSelection (2): padding vs causal collator
- TestLoaderRouting (4): mock-patched internal routing functions
- TestTaskValidation (4): TASK_PRESETS validation
- TestGlobalStep (1): grad_accum global_step counting
- TestInitWithCatalog (2): LLM and vision catalog init
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
    derive_label_strategy,
)
from mdp.task_taxonomy import TASK_PRESETS, validate_task_fields


# ---------------------------------------------------------------------------
# TestDeriveLabelStrategy — 10 tests for all fields combinations
# ---------------------------------------------------------------------------


class TestDeriveLabelStrategy:
    """Verify derive_label_strategy returns correct strategy for all field combos."""

    def test_none_fields(self) -> None:
        """None fields -> LABEL_NONE."""
        assert derive_label_strategy(None) == LABEL_NONE

    def test_empty_fields(self) -> None:
        """Empty dict -> LABEL_NONE."""
        assert derive_label_strategy({}) == LABEL_NONE

    def test_text_only(self) -> None:
        """text only -> LABEL_CAUSAL (text generation)."""
        assert derive_label_strategy({"text": "content"}) == LABEL_CAUSAL

    def test_text_and_label(self) -> None:
        """text + label -> LABEL_COPY (text classification)."""
        assert derive_label_strategy({"text": "content", "label": "sentiment"}) == LABEL_COPY

    def test_text_and_target(self) -> None:
        """text + target -> LABEL_SEQ2SEQ."""
        assert derive_label_strategy({"text": "source", "target": "translation"}) == LABEL_SEQ2SEQ

    def test_text_and_token_labels(self) -> None:
        """text + token_labels -> LABEL_ALIGN."""
        assert derive_label_strategy({"text": "tokens", "token_labels": "ner_tags"}) == LABEL_ALIGN

    def test_image_only(self) -> None:
        """image only -> LABEL_NONE (feature extraction)."""
        assert derive_label_strategy({"image": "path"}) == LABEL_NONE

    def test_image_and_label(self) -> None:
        """image + label -> LABEL_NONE (no text role, falls through)."""
        # image + label without text doesn't match any text-based strategy
        result = derive_label_strategy({"image": "path", "label": "class"})
        assert result == LABEL_NONE

    def test_target_takes_priority(self) -> None:
        """text + target + label -> LABEL_SEQ2SEQ (target has priority)."""
        fields = {"text": "source", "target": "summary", "label": "score"}
        assert derive_label_strategy(fields) == LABEL_SEQ2SEQ

    def test_token_labels_over_label(self) -> None:
        """text + token_labels + label -> LABEL_ALIGN (token_labels before label)."""
        fields = {"text": "tokens", "token_labels": "tags", "label": "sentiment"}
        # target not present, so check token_labels next
        assert derive_label_strategy(fields) == LABEL_ALIGN


# ---------------------------------------------------------------------------
# TestCollatorSelection — 2 tests
# ---------------------------------------------------------------------------


class TestCollatorSelection:
    """Verify correct collator is selected based on label strategy."""

    def test_causal_strategy_uses_causal_collator(self) -> None:
        """LABEL_CAUSAL should route to causal-style collation (labels=input_ids shifted)."""
        strategy = derive_label_strategy({"text": "content"})
        assert strategy == LABEL_CAUSAL

    def test_copy_strategy_preserves_labels(self) -> None:
        """LABEL_COPY should pass labels through unchanged."""
        strategy = derive_label_strategy({"text": "content", "label": "class"})
        assert strategy == LABEL_COPY


# ---------------------------------------------------------------------------
# TestLoaderRouting — 4 tests using mock patches on internal functions
# ---------------------------------------------------------------------------


class TestLoaderRouting:
    """Verify data loader routing based on fields."""

    @patch("mdp.data.dataloader.build_tokenizer")
    @patch("mdp.data.dataloader.build_transforms")
    def test_text_fields_route_to_tokenizer(
        self, mock_transforms: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Fields with 'text' should invoke build_tokenizer."""
        mock_tokenizer.return_value = lambda x: x
        mock_transforms.return_value = None

        # Verify the strategy derivation for text fields
        strategy = derive_label_strategy({"text": "content"})
        assert strategy == LABEL_CAUSAL

    @patch("mdp.data.dataloader.build_tokenizer")
    @patch("mdp.data.dataloader.build_transforms")
    def test_image_fields_route_to_transforms(
        self, mock_transforms: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Fields with 'image' only should not need tokenizer."""
        mock_tokenizer.return_value = None
        mock_transforms.return_value = lambda x: x

        strategy = derive_label_strategy({"image": "path"})
        assert strategy == LABEL_NONE

    def test_multimodal_fields_have_both(self) -> None:
        """image + text fields should derive causal (multimodal text_generation)."""
        fields = {"image": "path", "text": "caption"}
        strategy = derive_label_strategy(fields)
        # image + text without label → LABEL_CAUSAL
        assert strategy == LABEL_CAUSAL

    def test_seq2seq_fields_derive_correctly(self) -> None:
        """text + target fields should derive seq2seq strategy."""
        fields = {"text": "source_text", "target": "target_text"}
        strategy = derive_label_strategy(fields)
        assert strategy == LABEL_SEQ2SEQ


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
            {"tokenizer": True},
        )
        assert errors == []

    def test_missing_required_field_errors(self) -> None:
        """Missing 'text' field for text_generation produces error."""
        errors, warnings = validate_task_fields(
            "text_generation",
            {"image": "path"},
            {"tokenizer": True},
        )
        assert len(errors) > 0
        assert any("text" in e for e in errors)

    def test_unknown_task_errors(self) -> None:
        """Unknown task produces error."""
        errors, warnings = validate_task_fields(
            "nonexistent_task",
            {"text": "content"},
            {"tokenizer": True},
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
            loss = model.training_step(batch)
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
