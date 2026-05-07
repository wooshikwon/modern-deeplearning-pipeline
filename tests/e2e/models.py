"""Tiny model definitions for E2E tests.

All models implement BaseModel and are intentionally small (<1K params)
to keep tests fast and CPU-friendly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mdp.models.base import BaseModel


class TinyVisionModel(BaseModel):
    """Minimal CNN for classification tests.

    Architecture: Conv2d(3,8,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(1)
                  -> Flatten -> Linear(8,hidden_dim) [.classifier]
                  -> Linear(hidden_dim,num_classes) [.head]
    """

    _block_classes = None  # 반복 블록 없는 단순 CNN

    def __init__(self, num_classes: int = 2, hidden_dim: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(8, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = batch["pixel_values"]
        features = self.classifier(self.backbone(x))
        logits = self.head(F.relu(features))
        outputs = {"logits": logits, "features": features}
        if "labels" in batch:
            outputs["loss"] = F.cross_entropy(logits, batch["labels"])
        return outputs

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        outputs = self.forward(batch)
        labels = batch["labels"]
        loss = F.cross_entropy(outputs["logits"], labels)
        preds = outputs["logits"].argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
        return {"val_loss": loss.item(), "accuracy": accuracy}


class TinyFeatureMapModel(BaseModel):
    """Minimal CNN that outputs spatial feature maps.

    Architecture: Conv2d(3,out_channels,3,pad=1) -> ReLU as backbone.
    head = Identity (passthrough).

    forward returns feature maps with shape (B, C, H, W).
    """

    _block_classes = None  # 반복 블록 없는 단순 CNN

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.backbone = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Identity()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = batch["pixel_values"]
        features = self.backbone(x)
        logits = self.head(features)
        return {"logits": logits, "features": features, "loss": logits.mean()}

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        outputs = self.forward(batch)
        return {"val_loss": outputs["logits"].mean().item()}


class TinyLanguageModel(BaseModel):
    """Minimal Transformer LM for language generation tests.

    Architecture: Embedding + positional embedding
                  -> 1-layer TransformerEncoder
                  -> .lm_head = Linear(hidden_dim, vocab_size, bias=False)
    """

    _block_classes = None  # PyTorch TransformerEncoderLayer (테스트용, 커스텀 블록 없음)

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_heads: int = 2,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch["input_ids"]  # (B, L)
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=input_ids.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        logits = self.lm_head(x)  # (B, L, V)
        outputs = {"logits": logits}
        labels = batch.get("labels", batch.get("input_ids"))
        if labels is not None and labels.numel() > 0 and int(labels.max()) < logits.size(-1):
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            outputs["loss"] = F.cross_entropy(
                shifted_logits.view(-1, logits.size(-1)),
                shifted_labels.view(-1),
            )
        return outputs

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        outputs = self.forward(batch)
        logits = outputs["logits"][:, :-1].contiguous()
        labels = batch["input_ids"][:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size), labels.view(-1)
        )
        perplexity = math.exp(min(loss.item(), 20.0))  # clamp for stability
        return {"val_loss": loss.item(), "perplexity": perplexity}


class TinyTokenClassModel(BaseModel):
    """Minimal Transformer for token classification.

    Like TinyLanguageModel but .head = Linear(hidden_dim, num_classes).
    forward loss uses CE with ignore_index=-100.
    """

    _block_classes = None  # 반복 블록 없는 테스트용 모델

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 32,
        num_classes: int = 5,
        num_layers: int = 1,
        num_heads: int = 2,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch["input_ids"]  # (B, L)
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.transformer(x)
        logits = self.head(x)  # (B, L, C)
        outputs = {"logits": logits}
        if "labels" in batch:
            labels = batch["labels"]
            valid_labels = labels[labels != -100]
            if valid_labels.numel() == 0 or int(valid_labels.max()) < logits.size(-1):
                outputs["loss"] = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
        return outputs

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        outputs = self.forward(batch)
        logits = outputs["logits"]
        labels = batch["labels"]
        loss = F.cross_entropy(
            logits.view(-1, self.num_classes),
            labels.view(-1),
            ignore_index=-100,
        )
        return {"val_loss": loss.item()}


class TinyMoEModel(BaseModel):
    """Minimal MoE model for strategy tests.

    Architecture: Embedding -> 1-layer with MoE FFN
      - gate (router): Linear(hidden_dim, num_experts)
      - experts: ModuleList of Linear(hidden_dim, hidden_dim)
      - lm_head: Linear(hidden_dim, vocab_size)

    Mimics HuggingFace MoE structure (gate + experts ModuleList)
    so MoEStrategy hooks can detect and wrap it.
    """

    _block_classes = None  # 테스트용 MoE (실제 MoE 모델이면 DecoderLayer 지정)

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_dim: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
        max_len: int = 32,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # HuggingFace-style config for MoE detection
        self.config = type("Config", (), {
            "num_local_experts": num_experts,
            "num_experts_per_tok": top_k,
        })()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        # MoE layer — mimics MixtralDecoderLayer structure
        self.moe_layer = _TinyMoELayer(hidden_dim, num_experts, top_k)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch["input_ids"]
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.moe_layer(x)
        logits = self.lm_head(x)
        outputs = {"logits": logits}
        labels = batch.get("labels", batch.get("input_ids"))
        if labels is not None and labels.numel() > 0 and int(labels.max()) < logits.size(-1):
            outputs["loss"] = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
        return outputs

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        loss = self.forward(batch)["loss"]
        return {"val_loss": loss.item()}


class _TinyMoELayer(nn.Module):
    """MoE layer with gate + experts for testing.

    Structure matches what MoEStrategy._is_moe_layer() expects:
    a child named 'experts' that is an nn.ModuleList,
    and a child named 'gate' that is an nn.Linear.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Dense MoE forward (no EP, for single-process tests)."""
        B, L, H = hidden_states.shape
        flat = hidden_states.view(-1, H)
        router_logits = self.gate(flat)
        weights, indices = torch.topk(router_logits.softmax(dim=-1), k=self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e_idx in range(self.num_experts):
                mask = indices[:, k] == e_idx
                if mask.any():
                    output[mask] += weights[mask, k].unsqueeze(-1) * self.experts[e_idx](flat[mask])
        return output.view(B, L, H)


class TinyDualEncoderModel(BaseModel):
    """Minimal dual encoder for multimodal contrastive learning.

    image_encoder: Conv2d(3,hidden_dim,3,pad=1) -> AdaptiveAvgPool2d(1)
                   -> Flatten -> Linear(hidden_dim, projection_dim)
    text_encoder: Embedding(128, hidden_dim) -> mean pool
                  -> Linear(hidden_dim, projection_dim)
    """

    _block_classes = None  # 반복 블록 없는 테스트용 모델

    def __init__(
        self,
        hidden_dim: int = 16,
        projection_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.text_encoder = nn.Sequential(
            nn.Embedding(128, hidden_dim),
        )
        self.text_projection = nn.Linear(hidden_dim, projection_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        image_features = self.image_encoder(batch["pixel_values"])
        image_features = F.normalize(image_features, dim=-1)

        text_embeds = self.text_encoder(batch["input_ids"])  # (B, L, H)
        text_pooled = text_embeds.mean(dim=1)  # (B, H)
        text_features = self.text_projection(text_pooled)
        text_features = F.normalize(text_features, dim=-1)

        outputs = {
            "image_features": image_features,
            "text_features": text_features,
            "features": image_features,
        }
        logits = (image_features @ text_features.T) * self.temperature.exp()
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        outputs["loss"] = (loss_i2t + loss_t2i) / 2.0
        return outputs

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        loss = self.forward(batch)["loss"]
        return {"val_loss": loss.item()}
