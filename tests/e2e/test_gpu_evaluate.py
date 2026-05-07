"""Single-GPU fixture evaluation smoke tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.e2e.conftest import e2e_artifact_dir


class _Accuracy:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, outputs: dict, batch: dict) -> None:
        logits = outputs["logits"]
        labels = batch["labels"]
        preds = logits.argmax(dim=-1)
        self.correct += (preds == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> float:
        return self.correct / max(self.total, 1)


@pytest.mark.gpu
@pytest.mark.fixtures
def test_vit_fixture_evaluation_cuda(tmp_path, request, vit_tiny, cifar10_tiny):
    """ViT fixture runs evaluation metrics over the cached CIFAR tensor slice."""
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForImageClassification

    from mdp.callbacks.inference import DefaultOutputCallback
    from mdp.serving.inference import run_batch_inference

    samples = torch.load(cifar10_tiny, map_location="cpu")
    images = samples["images"][:4].float() / 255.0
    labels = samples["labels"][:4].long()

    model = AutoModelForImageClassification.from_pretrained(str(vit_tiny)).cuda().eval()
    image_size = int(model.config.image_size)
    images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)

    def _collate(batch):
        xs, ys = zip(*batch)
        return {"pixel_values": torch.stack(xs), "labels": torch.stack(ys)}

    loader = DataLoader(TensorDataset(images, labels), batch_size=2, collate_fn=_collate)
    run_dir = e2e_artifact_dir(tmp_path, request.node.name)
    output_path = run_dir / "vit-eval"
    output_cb = DefaultOutputCallback(output_path=output_path, output_format="jsonl", task="image_classification")
    result_path, eval_results = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=output_path,
        output_format="jsonl",
        task="image_classification",
        device="cuda",
        metrics=[_Accuracy()],
        callbacks=[output_cb],
    )

    assert result_path is not None and result_path.exists()
    assert "_Accuracy" in eval_results
    assert 0.0 <= eval_results["_Accuracy"] <= 1.0
