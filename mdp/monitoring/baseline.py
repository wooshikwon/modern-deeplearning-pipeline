"""Baseline computation and drift detection for data/model monitoring.

Computes statistical baselines from training data and model outputs,
then compares new distributions against baselines to detect drift.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Helpers ──


def _safe_log2(x: torch.Tensor) -> torch.Tensor:
    """Element-wise log2, clamping to avoid log(0)."""
    return torch.log2(x.clamp(min=1e-12))


def _jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions.

    Both p and q must be 1-D tensors summing to ~1.  Returns a scalar float
    in [0, 1] (using log base 2).
    """
    p = p.float().clamp(min=1e-12)
    q = q.float().clamp(min=1e-12)
    # Normalise just in case
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * (p * (_safe_log2(p) - _safe_log2(m))).sum() + 0.5 * (
        q * (_safe_log2(q) - _safe_log2(m))
    ).sum()
    return jsd.item()


# ── Baseline Computation ──


def compute_baseline(
    train_dataloader: Any,
    model: torch.nn.Module,
    config: Any | None = None,
    *,
    max_batches: int = 100,
) -> dict[str, Any]:
    """Compute a statistical baseline from training data and model outputs.

    Parameters
    ----------
    train_dataloader:
        An iterable yielding batch dicts (e.g. a ``torch.utils.data.DataLoader``).
    model:
        The model to run inference through for output statistics.
    config:
        Optional config object.  If it carries ``recipe.monitoring.baseline``
        with a ``max_batches`` key, that value overrides the keyword argument.
    max_batches:
        Maximum number of batches to sample.  Defaults to 100.

    Returns
    -------
    dict
        Baseline statistics including ``meta``, ``input_stats``, and
        ``output_stats`` sections.
    """
    # Allow config to override max_batches
    if config is not None:
        try:
            mb = config.recipe.monitoring.baseline.get("max_batches")
            if mb is not None:
                max_batches = int(mb)
        except Exception:
            pass

    device = next(model.parameters(), torch.tensor(0.0)).device

    # Accumulators -- input stats
    pixel_channel_sum: torch.Tensor | None = None
    pixel_channel_sq_sum: torch.Tensor | None = None
    pixel_count: int = 0

    token_lengths: list[int] = []

    label_counts: dict[int, int] = {}

    # Accumulators -- output stats
    entropies: list[float] = []
    confidences: list[float] = []

    num_samples = 0

    model.eval()

    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx >= max_batches:
            break

        # ── Input stats ──

        # Vision: pixel_values  [B, C, H, W]
        if "pixel_values" in batch:
            try:
                pv = batch["pixel_values"].float()
                b, c = pv.shape[0], pv.shape[1]
                if pixel_channel_sum is None:
                    pixel_channel_sum = torch.zeros(c, dtype=torch.float64)
                    pixel_channel_sq_sum = torch.zeros(c, dtype=torch.float64)
                # Reduce over B, H, W  -> [C]
                spatial = pv.reshape(b, c, -1)  # [B, C, H*W]
                n_pixels = spatial.shape[0] * spatial.shape[2]
                pixel_channel_sum += spatial.sum(dim=(0, 2)).double().cpu()
                pixel_channel_sq_sum += (spatial**2).sum(dim=(0, 2)).double().cpu()
                pixel_count += n_pixels
            except Exception:
                logger.debug("Failed to process pixel_values", exc_info=True)

        # Language: input_ids  [B, L]
        if "input_ids" in batch:
            try:
                ids = batch["input_ids"]
                # Compute non-padding lengths (assume pad_token_id=0 if not given)
                # Use attention_mask if available for accuracy
                if "attention_mask" in batch:
                    lengths = batch["attention_mask"].sum(dim=-1)
                else:
                    lengths = (ids != 0).sum(dim=-1)
                token_lengths.extend(lengths.tolist())
            except Exception:
                logger.debug("Failed to process input_ids", exc_info=True)

        # Labels
        if "labels" in batch:
            try:
                labs = batch["labels"].flatten()
                valid = labs[labs != -100]
                for v in valid.tolist():
                    label_counts[int(v)] = label_counts.get(int(v), 0) + 1
            except Exception:
                logger.debug("Failed to process labels", exc_info=True)

        # Count samples
        for key in ("pixel_values", "input_ids", "labels"):
            if key in batch:
                try:
                    num_samples += batch[key].shape[0]
                except Exception:
                    pass
                break

        # ── Output stats ──
        try:
            with torch.no_grad():
                # Move batch tensors to model device
                model_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_batch[k] = v.to(device)
                    else:
                        model_batch[k] = v

                outputs = model(**model_batch)

                # Extract logits
                logits = None
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]

                if logits is not None and logits.dim() >= 2:
                    # For sequence models [B, L, V], take last token or flatten
                    if logits.dim() == 3:
                        # Use all positions
                        logits_2d = logits.reshape(-1, logits.shape[-1])
                    else:
                        logits_2d = logits

                    probs = F.softmax(logits_2d.float(), dim=-1)
                    log_probs = _safe_log2(probs)
                    ent = -(probs * log_probs).sum(dim=-1)  # per-sample entropy
                    conf = probs.max(dim=-1).values

                    entropies.extend(ent.cpu().tolist())
                    confidences.extend(conf.cpu().tolist())
        except Exception:
            logger.debug("Failed to compute output stats for batch %d", batch_idx, exc_info=True)

    # ── Embedding centroids (opt-in) ──

    embedding_centroids: dict[str, list[float]] | None = None

    _ec_enabled = False
    if config is not None:
        try:
            _ec_enabled = bool(
                config.recipe.monitoring.baseline.get("embedding_centroids", False)
            )
        except Exception:
            pass

    if _ec_enabled:
        try:
            # 1. Find penultimate layer
            layers = list(model.children())
            target_layer = layers[-2] if len(layers) >= 2 else layers[-1]

            # 2. Register forward hook to capture activations
            embeddings_by_class: dict[int, list[torch.Tensor]] = defaultdict(list)
            hook_output: dict[str, torch.Tensor] = {}

            def _hook_fn(
                module: torch.nn.Module,
                input: Any,
                output: torch.Tensor,
            ) -> None:
                hook_output["features"] = output

            handle = target_layer.register_forward_hook(_hook_fn)

            # 3. Iterate over dataloader and collect per-class embeddings
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_dataloader):
                    if batch_idx >= max_batches:
                        break

                    # Move batch tensors to model device
                    model_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            model_batch[k] = v.to(device)
                        else:
                            model_batch[k] = v

                    model(**model_batch)

                    features = hook_output.get("features")
                    if features is None:
                        continue

                    # Sequence output (B, L, D) → mean pooling → (B, D)
                    if features.ndim == 3:
                        features = features.mean(dim=1)

                    labels = model_batch.get("labels")
                    if labels is None:
                        continue

                    # Flatten labels for per-sample assignment
                    if labels.ndim > 1:
                        # For sequence labelling, take the most common non-ignored label
                        # per sample as the "class" for centroid grouping.
                        per_sample_labels = []
                        for row in labels:
                            valid = row[row != -100]
                            if len(valid) > 0:
                                per_sample_labels.append(valid[0].item())
                            else:
                                per_sample_labels.append(None)
                    else:
                        per_sample_labels = labels.tolist()

                    for feat, label in zip(features, per_sample_labels):
                        if label is not None:
                            embeddings_by_class[int(label)].append(feat.cpu())

            # 4. Compute centroids
            centroids: dict[str, list[float]] = {}
            for cls, embs in embeddings_by_class.items():
                centroids[str(cls)] = torch.stack(embs).mean(dim=0).tolist()

            handle.remove()
            embedding_centroids = centroids
        except Exception:
            logger.debug("Failed to compute embedding centroids", exc_info=True)

    # ── Assemble result ──

    baseline: dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_samples": num_samples,
        },
        "input_stats": {},
        "output_stats": {},
    }

    # Vision input stats
    if pixel_channel_sum is not None and pixel_count > 0:
        mean = (pixel_channel_sum / pixel_count).tolist()
        var = (pixel_channel_sq_sum / pixel_count) - (pixel_channel_sum / pixel_count) ** 2
        std = var.clamp(min=0).sqrt().tolist()
        baseline["input_stats"]["vision"] = {
            "channel_mean": mean,
            "channel_std": std,
        }

    # Language input stats
    if token_lengths:
        tl = torch.tensor(token_lengths, dtype=torch.float64)
        sorted_tl = tl.sort().values
        n = len(sorted_tl)
        baseline["input_stats"]["language"] = {
            "token_length_mean": tl.mean().item(),
            "token_length_p50": sorted_tl[n // 2].item(),
            "token_length_p95": sorted_tl[min(int(n * 0.95), n - 1)].item(),
        }

    # Label distribution
    if label_counts:
        total = sum(label_counts.values())
        baseline["input_stats"]["label_distribution"] = {
            str(k): v / total for k, v in sorted(label_counts.items())
        }

    # Output stats
    if entropies:
        ent_t = torch.tensor(entropies, dtype=torch.float64)
        baseline["output_stats"]["entropy_mean"] = ent_t.mean().item()
        baseline["output_stats"]["entropy_std"] = ent_t.std().item() if len(entropies) > 1 else 0.0
    if confidences:
        conf_t = torch.tensor(confidences, dtype=torch.float64)
        baseline["output_stats"]["confidence_mean"] = conf_t.mean().item()

    # Embedding centroids (only present when opt-in is enabled and computation succeeded)
    if embedding_centroids is not None:
        baseline["embedding_centroids"] = embedding_centroids

    return baseline


# ── Drift Comparison ──


def compare_baselines(
    baseline: dict[str, Any],
    current: dict[str, Any],
    config: Any | None = None,
) -> dict[str, Any]:
    """Compare a current measurement against a stored baseline for drift.

    Parameters
    ----------
    baseline:
        Reference baseline dict produced by :func:`compute_baseline`.
    current:
        New measurement dict with the same structure.
    config:
        Optional config.  If ``recipe.monitoring.drift`` exists, keys
        ``entropy_threshold`` (default 2.0) and ``jsd_threshold``
        (default 0.1) are read from it.

    Returns
    -------
    dict
        ``{"drift_detected": bool, "drift_score": float, "alerts": [str]}``
    """
    # Defaults
    entropy_threshold: float = 2.0
    jsd_threshold: float = 0.1

    if config is not None:
        try:
            drift_cfg = config.recipe.monitoring.drift
            if isinstance(drift_cfg, dict):
                entropy_threshold = float(drift_cfg.get("entropy_threshold", entropy_threshold))
                jsd_threshold = float(drift_cfg.get("jsd_threshold", jsd_threshold))
        except Exception:
            pass

    alerts: list[str] = []
    drift_scores: list[float] = []

    # ── Entropy drift ──
    b_out = baseline.get("output_stats", {})
    c_out = current.get("output_stats", {})

    if "entropy_mean" in b_out and "entropy_mean" in c_out:
        b_mean = b_out["entropy_mean"]
        b_std = b_out.get("entropy_std", 1.0)
        c_mean = c_out["entropy_mean"]

        if b_std <= 0:
            b_std = 1.0  # avoid division by zero

        entropy_diff = abs(c_mean - b_mean)
        entropy_drifted = entropy_diff > entropy_threshold * b_std

        normalised_score = entropy_diff / b_std if b_std > 0 else 0.0
        drift_scores.append(normalised_score)

        if entropy_drifted:
            alerts.append(
                f"Entropy drift detected: |{c_mean:.4f} - {b_mean:.4f}| = {entropy_diff:.4f} "
                f"> {entropy_threshold} * {b_std:.4f}"
            )

    # ── Class distribution drift (JSD) ──
    b_input = baseline.get("input_stats", {})
    c_input = current.get("input_stats", {})

    b_dist = b_input.get("label_distribution", {})
    c_dist = c_input.get("label_distribution", {})

    if b_dist and c_dist:
        # Align distributions on union of keys
        all_keys = sorted(set(b_dist.keys()) | set(c_dist.keys()))
        p = torch.tensor([float(b_dist.get(k, 0.0)) for k in all_keys])
        q = torch.tensor([float(c_dist.get(k, 0.0)) for k in all_keys])

        # Normalise
        if p.sum() > 0:
            p = p / p.sum()
        if q.sum() > 0:
            q = q / q.sum()

        jsd = _jensen_shannon_divergence(p, q)
        drift_scores.append(jsd / jsd_threshold if jsd_threshold > 0 else 0.0)

        if jsd > jsd_threshold:
            alerts.append(
                f"Class distribution drift detected: JSD = {jsd:.6f} > {jsd_threshold}"
            )

    # ── Embedding centroid drift (cosine distance) ──
    embedding_drift: dict[str, Any] | None = None

    b_centroids = baseline.get("embedding_centroids", {})
    c_centroids = current.get("embedding_centroids", {})

    if b_centroids and c_centroids:
        embedding_drift_threshold: float = 0.1
        if config is not None:
            try:
                drift_cfg = config.recipe.monitoring.drift
                if isinstance(drift_cfg, dict):
                    embedding_drift_threshold = float(
                        drift_cfg.get("embedding_drift_threshold", embedding_drift_threshold)
                    )
            except Exception:
                pass

        common_classes = set(b_centroids.keys()) & set(c_centroids.keys())
        distances: list[float] = []

        for cls in common_classes:
            b_vec = torch.tensor(b_centroids[cls], dtype=torch.float32)
            c_vec = torch.tensor(c_centroids[cls], dtype=torch.float32)
            cos_dist = 1.0 - F.cosine_similarity(
                b_vec.unsqueeze(0), c_vec.unsqueeze(0),
            ).item()
            distances.append(cos_dist)

        if distances:
            mean_dist = sum(distances) / len(distances)
            embedding_drift = {
                "mean_cosine_distance": mean_dist,
                "drift_detected": mean_dist > embedding_drift_threshold,
                "num_classes_compared": len(common_classes),
            }
            drift_scores.append(mean_dist / embedding_drift_threshold if embedding_drift_threshold > 0 else 0.0)

            if mean_dist > embedding_drift_threshold:
                alerts.append(
                    f"Embedding drift: cosine distance {mean_dist:.4f} > {embedding_drift_threshold}"
                )

    # Aggregate
    drift_score = max(drift_scores) if drift_scores else 0.0
    drift_detected = len(alerts) > 0

    report: dict[str, Any] = {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "alerts": alerts,
    }
    if embedding_drift is not None:
        report["embedding_drift"] = embedding_drift

    return report
