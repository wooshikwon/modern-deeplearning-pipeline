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
        return {"logits": logits, "features": features}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.forward(batch)
        labels = batch["labels"]
        return F.cross_entropy(outputs["logits"], labels)

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
        return {"logits": logits, "features": features}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.forward(batch)
        return outputs["logits"].mean()

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        outputs = self.forward(batch)
        return {"val_loss": outputs["logits"].mean().item()}


class TinyLanguageModel(BaseModel):
    """Minimal Transformer LM for language generation tests.

    Architecture: Embedding + positional embedding
                  -> 1-layer TransformerEncoder
                  -> .lm_head = Linear(hidden_dim, vocab_size, bias=False)
    """

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
        return {"logits": logits}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.forward(batch)
        logits = outputs["logits"][:, :-1].contiguous()  # shifted
        labels = batch["input_ids"][:, 1:].contiguous()
        return F.cross_entropy(
            logits.view(-1, self.vocab_size), labels.view(-1)
        )

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
    training_step uses CE with ignore_index=-100.
    """

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
        return {"logits": logits}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.forward(batch)
        logits = outputs["logits"]
        labels = batch["labels"]  # (B, L)
        return F.cross_entropy(
            logits.view(-1, self.num_classes),
            labels.view(-1),
            ignore_index=-100,
        )

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


class TinyDualEncoderModel(BaseModel):
    """Minimal dual encoder for multimodal contrastive learning.

    image_encoder: Conv2d(3,hidden_dim,3,pad=1) -> AdaptiveAvgPool2d(1)
                   -> Flatten -> Linear(hidden_dim, projection_dim)
    text_encoder: Embedding(128, hidden_dim) -> mean pool
                  -> Linear(hidden_dim, projection_dim)
    """

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

        return {
            "image_features": image_features,
            "text_features": text_features,
            "features": image_features,
        }

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.forward(batch)
        img_f = outputs["image_features"]
        txt_f = outputs["text_features"]

        # Symmetric contrastive loss (CLIP-style)
        logits = (img_f @ txt_f.T) * self.temperature.exp()
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        loss = self.training_step(batch)
        return {"val_loss": loss.item()}
