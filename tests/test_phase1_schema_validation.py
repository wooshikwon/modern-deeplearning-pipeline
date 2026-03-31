"""Phase 1 완료 검증: 03 스키마 예제 5개 YAML을 SettingsFactory로 파싱."""

from pathlib import Path

from mdp.settings.factory import SettingsFactory

FIXTURES = Path(__file__).parent / "fixtures"

EXAMPLES = [
    {
        "label": "1. ViT LoRA (local, single GPU)",
        "recipe": "recipes/vit-lora-cifar10.yaml",
        "config": "configs/local-single-gpu.yaml",
    },
    {
        "label": "2. GPT-2 DDP (remote, 4 GPU)",
        "recipe": "recipes/gpt2-finetune-text.yaml",
        "config": "configs/remote-4gpu-ddp.yaml",
    },
    {
        "label": "3. YOLO Detection (local, single GPU)",
        "recipe": "recipes/yolo-detection-custom.yaml",
        "config": "configs/local-single-gpu-detection.yaml",
    },
    {
        "label": "4. Qwen QLoRA (multi-node, DeepSpeed)",
        "recipe": "recipes/qwen25-qlora-instruct.yaml",
        "config": "configs/multi-node-2x4gpu-deepspeed.yaml",
    },
    {
        "label": "5. CLIP (cloud, GCP 8 GPU)",
        "recipe": "recipes/clip-finetune-custom.yaml",
        "config": "configs/cloud-gcp-8gpu.yaml",
    },
]


def main() -> None:
    factory = SettingsFactory()
    passed = 0
    failed = 0

    for ex in EXAMPLES:
        label = ex["label"]
        recipe_path = str(FIXTURES / ex["recipe"])
        config_path = str(FIXTURES / ex["config"])

        try:
            settings = factory.for_training(recipe_path, config_path)
            r = settings.recipe
            c = settings.config
            print(f"[PASS] {label}")
            print(f"       name={r.name}, task={r.task}")
            print(f"       model.class_path={r.model.class_path}")
            print(f"       compute.target={c.compute.target}")
            if r.adapter:
                print(f"       adapter.method={r.adapter.method}")
            print(f"       training.epochs={r.training.epochs}, precision={r.training.precision}")
            print()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {label}")
            print(f"       Error: {e}")
            print()
            failed += 1

    print(f"Result: {passed}/5 passed, {failed}/5 failed")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
