"""E2E tests for Factory._attach_head() with _target_attr routing.

Tests verify head attachment to specific model attributes, error handling
for missing or nonexistent attributes, and alternative target attributes.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from mdp.factory.factory import Factory
from mdp.models.heads.classification import ClassificationHead
from tests.e2e.models import TinyVisionModel


class TestAttachHead:
    """Factory._attach_head() end-to-end tests."""

    def test_attach_head_with_target_attr(self) -> None:
        """Attach ClassificationHead to TinyVisionModel.head via _target_attr='head'."""
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        original_head = model.head

        new_head = ClassificationHead(num_classes=10, hidden_dim=16, dropout=0.0)
        Factory._attach_head(model, new_head, target_attr="head")

        assert model.head is new_head
        assert model.head is not original_head

    def test_attach_head_wrong_attr_raises(self) -> None:
        """target_attr='nonexistent' should raise AttributeError."""
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        new_head = ClassificationHead(num_classes=10, hidden_dim=16, dropout=0.0)

        with pytest.raises(AttributeError, match="nonexistent"):
            Factory._attach_head(model, new_head, target_attr="nonexistent")

    def test_attach_head_missing_target_attr_raises(self) -> None:
        """target_attr=None should raise ValueError."""
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        new_head = ClassificationHead(num_classes=10, hidden_dim=16, dropout=0.0)

        with pytest.raises(ValueError, match="_target_attr"):
            Factory._attach_head(model, new_head, target_attr=None)

    def test_attach_head_to_classifier_attr(self) -> None:
        """Attach head to 'classifier' attribute (TinyVisionModel has .classifier)."""
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        original_classifier = model.classifier

        # classifier is Linear(8, hidden_dim), replace with a new head
        new_head = nn.Linear(8, 32)
        Factory._attach_head(model, new_head, target_attr="classifier")

        assert model.classifier is new_head
        assert model.classifier is not original_classifier
