"""Padding ratio sanity script — length-bucketed sampler 효과 실측.

spec-length-bucketed-sampler.md U6의 sanity 측정 도구. Recipe 하나를 입력으로
받아 일정 step만큼 DataLoader iter를 돌리며 batch마다 ``attention_mask``의
sample-별 valid 토큰 수를 집계, **batch_max - batch_mean** 차이가 batch_max에서
차지하는 비율(= padding 비율)을 계산한다.

Random shuffle 기준선과 length-bucketed sampler를 한 스크립트로 비교하기 위해
``--sampler-config``로 sampler를 옵션 주입할 수 있다 (Recipe.data.sampler가
이미 지정되어 있으면 override).

사용 예::

    # baseline (sampler 미지정 → recipe 그대로)
    uv run python scripts/measure_padding_ratio.py \\
        --recipe recipes/wntp_baseline.yaml --num-steps 100

    # length-bucketed sampler로 override
    uv run python scripts/measure_padding_ratio.py \\
        --recipe recipes/wntp_baseline.yaml --num-steps 100 \\
        --sampler-config '{"_component_": "LengthGroupedBatchSampler", "bucket_size": 256}'

비-범위:
    실제 GPU 학습 시간(step time) 측정은 본 스크립트의 책임이 아니다 —
    weighted-ntp 측 후속 작업으로 vast.ai에서 실측한다. 본 스크립트는 padding
    비율만 측정하여 sampler 선택의 정량 효과를 빠르게 검증한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 사용자가 ``mdp.tests.fixtures`` 류 외부 모듈을 _component_로 주입할 수 있도록
# 스크립트 진입 시점에 현재 작업 디렉토리를 sys.path에 우선 추가한다 — 본 스크립트는
# 학습 entrypoint가 아니라 sanity 도구이므로 import 환경을 명시적으로 확장한다.
_CWD = str(Path.cwd().resolve())
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

import torch  # noqa: E402

from mdp.data.dataloader import create_dataloaders  # noqa: E402
from mdp.settings.loader import SettingsLoader  # noqa: E402


def _load_recipe_settings(recipe_path: str) -> Any:
    """Recipe 단독 로드. config 페어가 없어도 ``for_estimation``으로 진입 가능."""
    return SettingsLoader().load_estimation_settings(recipe_path)


def _override_sampler(recipe_data: Any, sampler_config: dict[str, Any] | None) -> None:
    """Recipe.data.sampler를 in-place로 override (None이면 no-op)."""
    if sampler_config is not None:
        recipe_data.sampler = sampler_config


def _per_sample_lengths(batch: Any) -> torch.Tensor | None:
    """batch에서 sample-별 유효 토큰 수를 추출한다.

    우선순위:
    1. ``attention_mask`` (dict[str, Tensor]) → ``sum(-1)``
    2. ``input_ids`` (dict[str, Tensor]) → 각 행의 길이 (padding이 없는 경우)
    3. list-of-dict → ``length`` 키 또는 ``input_ids`` 길이

    None을 반환하면 호출자는 해당 batch를 skip한다.

    Note — ``input_ids`` 형태 가정:
        분기 2는 ``input_ids``가 ``(B, S)`` 2D 텐서임을 가정하고 ``shape[-1]``을
        시퀀스 길이로 사용한다. packed sequences ``(B,)`` 1D나 sparse 표현 등은
        잘못된 길이를 반환한다. 본 spec(length-bucketed sampler)은 sequence
        packing을 명시적 비범위로 두므로 현재 위험은 없다 — packing 도입 시
        본 분기를 재검토하거나 측정 대상별 length 추출 로직을 외부 hook으로
        주입할 수 있게 확장할 것.
    """
    if isinstance(batch, dict):
        attn = batch.get("attention_mask")
        if isinstance(attn, torch.Tensor):
            return attn.sum(dim=-1).to(dtype=torch.float32)
        ids = batch.get("input_ids")
        if isinstance(ids, torch.Tensor):
            # padding 없는 경우(텐서가 (B, S) 형태) 모든 sample이 S 토큰으로 간주
            return torch.full(
                (ids.shape[0],), float(ids.shape[-1]), dtype=torch.float32
            )
        # length 키가 직접 있는 경우 (fixture/test용)
        ln = batch.get("length")
        if isinstance(ln, torch.Tensor):
            return ln.to(dtype=torch.float32)
    elif isinstance(batch, list) and batch:
        # list-of-dict (collator가 stack을 안 한 경우)
        first = batch[0]
        if isinstance(first, dict):
            if "length" in first:
                return torch.tensor(
                    [float(item["length"]) for item in batch], dtype=torch.float32
                )
            if "input_ids" in first:
                return torch.tensor(
                    [float(len(item["input_ids"])) for item in batch],
                    dtype=torch.float32,
                )
    return None


def measure(
    recipe_path: str,
    sampler_config: dict[str, Any] | None,
    num_steps: int,
) -> dict[str, float]:
    """Recipe 기반 DataLoader를 ``num_steps``만큼 돌며 padding ratio를 계산한다.

    Returns:
        ``{"avg_batch_mean": float, "avg_batch_max": float, "padding_ratio_pct": float,
           "num_batches": int}``.
        측정 batch가 0건이면 모든 값이 0.0이고 ``num_batches=0``.
    """
    settings = _load_recipe_settings(recipe_path)
    recipe = settings.recipe
    _override_sampler(recipe.data, sampler_config)

    distributed = settings.config.compute.distributed is not None
    loaders = create_dataloaders(
        dataset_config=recipe.data.dataset,
        collator_config=recipe.data.collator,
        dataloader_config=recipe.data.dataloader.model_dump(),
        val_dataset_config=recipe.data.val_dataset,
        sampler_config=recipe.data.sampler,
        distributed=distributed,
    )
    train = loaders["train"]

    batch_means: list[float] = []
    batch_maxes: list[float] = []
    skipped = 0

    for i, batch in enumerate(train):
        if i >= num_steps:
            break
        per_sample = _per_sample_lengths(batch)
        if per_sample is None or per_sample.numel() == 0:
            skipped += 1
            continue
        batch_means.append(float(per_sample.mean().item()))
        batch_maxes.append(float(per_sample.max().item()))

    if not batch_means:
        return {
            "avg_batch_mean": 0.0,
            "avg_batch_max": 0.0,
            "padding_ratio_pct": 0.0,
            "num_batches": 0,
            "num_skipped": float(skipped),
        }

    avg_mean = sum(batch_means) / len(batch_means)
    avg_max = sum(batch_maxes) / len(batch_maxes)
    pad_ratio = ((avg_max - avg_mean) / avg_max * 100.0) if avg_max > 0 else 0.0
    return {
        "avg_batch_mean": avg_mean,
        "avg_batch_max": avg_max,
        "padding_ratio_pct": pad_ratio,
        "num_batches": len(batch_means),
        "num_skipped": float(skipped),
    }


def _format_sampler_label(sampler_config: dict[str, Any] | None) -> str:
    if sampler_config is None:
        return "<none — recipe default (random shuffle or DistributedSampler)>"
    name = sampler_config.get("_component_", "<unknown>")
    extras = {k: v for k, v in sampler_config.items() if k != "_component_"}
    if extras:
        return f"{name} {extras}"
    return str(name)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Length-bucketed sampler의 padding ratio sanity 측정",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        type=str,
        help="Recipe YAML 경로",
    )
    parser.add_argument(
        "--sampler-config",
        type=str,
        default=None,
        help="JSON-encoded sampler config dict (Recipe.data.sampler를 override). "
        '예: \'{"_component_": "LengthGroupedBatchSampler", "bucket_size": 256}\'',
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="iterate할 batch 개수",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    sampler_config: dict[str, Any] | None = None
    if args.sampler_config:
        try:
            sampler_config = json.loads(args.sampler_config)
        except json.JSONDecodeError as exc:
            print(
                f"[ERROR] --sampler-config JSON 파싱 실패: {exc}",
                file=sys.stderr,
            )
            return 2
        if not isinstance(sampler_config, dict):
            print(
                "[ERROR] --sampler-config는 JSON object여야 한다",
                file=sys.stderr,
            )
            return 2

    recipe_label = Path(args.recipe).name
    sampler_label = _format_sampler_label(sampler_config)

    result = measure(args.recipe, sampler_config, args.num_steps)

    print(f"Recipe: {recipe_label}")
    print(f"Sampler: {sampler_label}")
    print(f"Steps: {result['num_batches']} (requested {args.num_steps})")
    if result["num_batches"] == 0:
        print("[WARN] 측정 가능한 batch 0건 — attention_mask/input_ids/length를 추출할 수 없었다")
        return 1
    print(f"Avg batch_mean: {result['avg_batch_mean']:.1f} tokens")
    print(f"Avg batch_max:  {result['avg_batch_max']:.1f} tokens")
    print(f"Padding ratio:  {result['padding_ratio_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
