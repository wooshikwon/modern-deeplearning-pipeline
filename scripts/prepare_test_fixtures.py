"""Cache small but real HuggingFace models + small dataset slices for multi-GPU e2e tests.

Designed to run on a remote cloud instance after `cloud_provision.sh`.
Caches into /workspace/test-fixtures/ (or --target-dir).

Model selection criteria:
  - Genuinely trained (not random-init) so forward outputs and tokenizer have meaning.
  - Small enough that all 5 fit under ~1GB (RTX 3090 24GB GPU has plenty of room).
  - Already published in safetensors format on HF Hub — no .bin conversion needed.
  - Each repo bundles its matching tokenizer (or no tokenizer for vision-only).

Usage:
  python scripts/prepare_test_fixtures.py
  python scripts/prepare_test_fixtures.py --target-dir /workspace/test-fixtures
  python scripts/prepare_test_fixtures.py --skip-datasets
  python scripts/prepare_test_fixtures.py --clean

Outputs (target-dir layout):
  models/{smollm2,gpt2,bert-tiny,vit-tiny}/              # HF snapshots (safetensors)
  data/wikitext-2-tiny/train.jsonl                       # 1k rows
  data/cifar10-tiny/samples.pt                           # 1k samples
  data/{preference-tiny,prompt-tiny,classification-text-tiny}/train.jsonl
  manifest.json                                          # arch + n_params + paths

Verification:
  Each model is loaded via AutoModelForX.from_pretrained() AND a forward pass
  is run on a synthetic batch to confirm the snapshot is fully usable downstream.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Real, small, safetensors-native models. Each entry: (local name, HF repo,
# task family). Family steers verify_models forward signature.
#
# Multimodal (CLIP) is intentionally absent here — the canonical small CLIP
# repos still ship .bin only, and a thorough CLIP e2e is deferred to the
# next coverage spec along with DPO/GRPO/FSDP/eval extensions.
TINY_MODELS: list[tuple[str, str, str]] = [
    ("smollm2",   "HuggingFaceTB/SmolLM2-135M",            "causal-lm"),
    ("gpt2",      "gpt2",                                   "causal-lm"),
    ("bert-tiny", "google/bert_uncased_L-2_H-128_A-2",      "encoder"),
    ("vit-tiny",  "WinKawaks/vit-tiny-patch16-224",         "vision"),
]


def cache_models(models_dir: Path) -> dict[str, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    out: dict[str, dict[str, Any]] = {}
    for local_name, repo_id, family in TINY_MODELS:
        target = models_dir / local_name
        # Sentinel: any of the standard weight file names indicates a complete snapshot.
        has_weights = (
            (target / "model.safetensors").exists()
            or any(target.glob("model-*.safetensors"))
        )
        if has_weights:
            print(f"  [skip] {local_name} already cached at {target}")
        else:
            print(f"  [pull] {repo_id} -> {target}")
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                # Skip .bin variants — these repos all ship safetensors.
                allow_patterns=[
                    "*.safetensors", "*.json", "*.txt", "*.model",
                    "tokenizer*", "vocab*", "merges*", "special_tokens*",
                    "preprocessor*",
                ],
            )
        size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
        out[local_name] = {
            "repo_id": repo_id,
            "family": family,
            "path": str(target),
            "size_mb": round(size_mb, 2),
        }
    return out


def verify_models(cached: dict[str, dict[str, Any]]) -> None:
    """Re-load each cached model with AutoModelForX AND run a forward pass.

    This is the actual operability check — if torch can't load the weights or
    the model+tokenizer pair is inconsistent, we fail here instead of mid-pytest.
    """
    print("verifying model load + forward pass...")
    import torch
    from transformers import (
        AutoConfig, AutoTokenizer,
        AutoModelForCausalLM, AutoModelForSequenceClassification,
        AutoModelForImageClassification,
    )

    head_for_family = {
        "causal-lm": AutoModelForCausalLM,
        "encoder": AutoModelForSequenceClassification,
        "vision": AutoModelForImageClassification,
    }

    for name, info in cached.items():
        path = info["path"]
        family = info["family"]
        try:
            config = AutoConfig.from_pretrained(path)
            cls = head_for_family[family]
            model = cls.from_pretrained(path)
            n_params = sum(p.numel() for p in model.parameters())

            with torch.no_grad():
                if family == "causal-lm":
                    tok = AutoTokenizer.from_pretrained(path)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    inputs = tok(["hello world"], return_tensors="pt", padding=True, truncation=True, max_length=8)
                    out = model(**inputs)
                    assert out.logits.shape[0] == 1
                elif family == "encoder":
                    tok = AutoTokenizer.from_pretrained(path)
                    inputs = tok(["a sentence"], return_tensors="pt", padding=True, truncation=True, max_length=8)
                    out = model(**inputs)
                    assert out.logits.shape[0] == 1
                elif family == "vision":
                    image_size = config.image_size
                    out = model(pixel_values=torch.randn(1, 3, image_size, image_size))
                    assert out.logits.shape[0] == 1
            info["n_params"] = n_params
            info["arch"] = config.architectures[0] if config.architectures else cls.__name__
            print(f"  ok: {name} ({info['arch']}, {n_params:,} params, forward ok)")
            del model
        except Exception as e:
            info["error"] = str(e)
            print(f"  FAIL: {name} — {e}", file=sys.stderr)


# ────────────────────────────────────────────────────────────
# Datasets
# ────────────────────────────────────────────────────────────

DATASETS = {
    "wikitext-2-tiny": {
        "hf_name": "wikitext", "config": "wikitext-2-raw-v1",
        "split": "train", "n_rows": 1000,
    },
    "cifar10-tiny": {
        "hf_name": "cifar10", "config": None,
        "split": "train", "n_rows": 1000,
    },
    "preference-tiny": {"kind": "synthetic-jsonl", "n_rows": 16},
    "prompt-tiny": {"kind": "synthetic-jsonl", "n_rows": 16},
    "classification-text-tiny": {"kind": "synthetic-jsonl", "n_rows": 16},
}


def cache_dataset_slice(data_dir: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    target = data_dir / name
    target.mkdir(parents=True, exist_ok=True)
    sentinel = target / ".cached"
    if sentinel.exists():
        print(f"  [skip] {name} already cached")
        return {"path": str(target), "n_rows": spec["n_rows"], "cached": True}

    if spec.get("kind") == "synthetic-jsonl":
        out_path = target / "train.jsonl"
        print(f"  [write] synthetic {name} -> {out_path}")
        with out_path.open("w") as f:
            for i in range(spec["n_rows"]):
                if name == "preference-tiny":
                    row = {
                        "chosen": f"Question {i}: answer with concise helpful detail.",
                        "rejected": f"Question {i}: irrelevant response.",
                    }
                elif name == "prompt-tiny":
                    row = {"text": f"Write one useful sentence about fixture prompt {i}."}
                elif name == "classification-text-tiny":
                    row = {"text": f"tiny classification example {i}", "label": i % 2}
                else:
                    raise ValueError(f"unknown synthetic dataset: {name}")
                f.write(json.dumps(row) + "\n")
        sentinel.touch()
        size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
        return {"path": str(target), "n_rows": spec["n_rows"], "size_mb": round(size_mb, 2)}

    print(f"  [pull] {spec['hf_name']} -> {target}")
    from datasets import load_dataset

    if spec.get("config"):
        ds = load_dataset(spec["hf_name"], spec["config"], split=f"{spec['split']}[:{spec['n_rows']}]")
    else:
        ds = load_dataset(spec["hf_name"], split=f"{spec['split']}[:{spec['n_rows']}]")

    if name == "wikitext-2-tiny":
        out_path = target / "train.jsonl"
        with out_path.open("w") as f:
            for row in ds:
                f.write(json.dumps({"text": row["text"]}) + "\n")
    elif name == "cifar10-tiny":
        import torch
        images, labels = [], []
        for row in ds:
            img = row["img"] if "img" in ds.column_names else row["image"]
            arr = torch.tensor(list(img.getdata()), dtype=torch.uint8) \
                .reshape(img.size[1], img.size[0], 3).permute(2, 0, 1)
            images.append(arr)
            labels.append(row["label"])
        torch.save(
            {"images": torch.stack(images), "labels": torch.tensor(labels)},
            target / "samples.pt",
        )
    else:
        ds.save_to_disk(str(target))

    sentinel.touch()
    size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
    return {"path": str(target), "n_rows": spec["n_rows"], "size_mb": round(size_mb, 2)}


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default="/workspace/test-fixtures")
    ap.add_argument("--skip-datasets", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete target-dir before fetching")
    args = ap.parse_args()

    target = Path(args.target_dir).expanduser().resolve()
    if args.clean and target.exists():
        print(f"cleaning {target}...")
        shutil.rmtree(target)

    models_dir = target / "models"
    data_dir = target / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"target_dir": str(target), "models": {}, "datasets": {}}

    print(f"caching models -> {models_dir}")
    manifest["models"] = cache_models(models_dir)
    if not args.skip_verify:
        verify_models(manifest["models"])

    if not args.skip_datasets:
        print(f"caching dataset slices -> {data_dir}")
        for name, spec in DATASETS.items():
            try:
                manifest["datasets"][name] = cache_dataset_slice(data_dir, name, spec)
            except Exception as e:
                manifest["datasets"][name] = {"error": str(e)}
                print(f"  FAIL: {name} — {e}", file=sys.stderr)

    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest written: {manifest_path}")

    total_models_mb = sum(v.get("size_mb", 0) for v in manifest["models"].values())
    total_data_mb = sum(v.get("size_mb", 0) for v in manifest["datasets"].values() if isinstance(v, dict))
    print(f"total: models={total_models_mb:.1f}MB, data={total_data_mb:.1f}MB")

    failed = [k for k, v in manifest["models"].items() if "error" in v]
    failed += [k for k, v in manifest["datasets"].items() if isinstance(v, dict) and "error" in v]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1

    print("FIXTURES_READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
