"""Serving/deployment artifact writer boundary."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Any, Literal

from mdp.artifacts.layout import WeightLayout, detect_weight_layout

logger = logging.getLogger(__name__)

ServingArtifactMode = Literal[
    "mlflow_snapshot",
    "deployment_export",
    "custom_export",
]


@dataclass(frozen=True)
class ServingArtifactRecord:
    output_dir: Path
    mode: ServingArtifactMode
    weight_layout: WeightLayout
    recipe_file: Path | None = None


class ServingArtifactManager:
    """Owns serving/export artifact write layouts."""

    def write(
        self,
        model: Any,
        settings: Any,
        output_dir: Path,
        *,
        mode: ServingArtifactMode,
        policy_state_dict: dict | None = None,
        recipe_source_dir: Path | None = None,
    ) -> ServingArtifactRecord:
        output_dir.mkdir(parents=True, exist_ok=True)
        if mode == "mlflow_snapshot":
            self._write_mlflow_snapshot(
                model,
                settings,
                output_dir,
                policy_state_dict=policy_state_dict,
            )
        elif mode in {"deployment_export", "custom_export"}:
            self._write_deployment_export(model, output_dir)
        else:
            raise ValueError(f"지원하지 않는 serving artifact mode: {mode}")

        recipe_file = self._write_recipe(settings, output_dir, recipe_source_dir)
        self._write_tokenizer(settings, output_dir)
        return ServingArtifactRecord(
            output_dir=output_dir,
            mode=mode,
            weight_layout=detect_weight_layout(output_dir),
            recipe_file=recipe_file,
        )

    def _write_mlflow_snapshot(
        self,
        model: Any,
        settings: Any,
        output_dir: Path,
        *,
        policy_state_dict: dict | None,
    ) -> None:
        target = getattr(model, "module", model)
        has_adapter = hasattr(target, "peft_config")

        if policy_state_dict is not None:
            if has_adapter:
                from peft import get_peft_model_state_dict
                from safetensors.torch import save_file

                adapter_names = list(target.peft_config.keys())
                adapter_name = adapter_names[0] if adapter_names else "default"
                adapter_sd = get_peft_model_state_dict(
                    target,
                    state_dict=policy_state_dict,
                    adapter_name=adapter_name,
                )
                save_file(adapter_sd, str(output_dir / "adapter_model.safetensors"))
                for _adapter_name, cfg in target.peft_config.items():
                    cfg.save_pretrained(str(output_dir))
            else:
                from safetensors.torch import save_file

                save_file(policy_state_dict, str(output_dir / "model.safetensors"))
        elif has_adapter or hasattr(target, "save_pretrained"):
            target.save_pretrained(output_dir)
        else:
            from safetensors.torch import save_file

            save_file(target.state_dict(), output_dir / "model.safetensors")

    def _write_deployment_export(self, model: Any, output_dir: Path) -> None:
        target = getattr(model, "module", model)
        if hasattr(target, "export"):
            target.export(output_dir)
        elif hasattr(target, "save_pretrained"):
            target.save_pretrained(output_dir)
        else:
            from safetensors.torch import save_file

            save_file(target.state_dict(), output_dir / "model.safetensors")

    def _write_recipe(
        self,
        settings: Any,
        output_dir: Path,
        recipe_source_dir: Path | None,
    ) -> Path | None:
        import yaml

        recipe_target = output_dir / "recipe.yaml"
        if recipe_source_dir is not None:
            recipe_src = recipe_source_dir / "recipe.yaml"
            if recipe_src.exists():
                shutil.copy(recipe_src, recipe_target)
                return recipe_target

        recipe = getattr(settings, "recipe", None)
        if recipe is None:
            return None
        recipe_dict = recipe.model_dump(mode="json")
        recipe_target.write_text(yaml.dump(recipe_dict, allow_unicode=True))
        return recipe_target

    def _write_tokenizer(self, settings: Any, output_dir: Path) -> None:
        recipe = getattr(settings, "recipe", None)
        if recipe is None:
            return
        from mdp.settings.components import component_kwargs

        tokenizer_name = component_kwargs(recipe.data.collator).get("tokenizer")
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer

                AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_dir)
            except Exception as e:
                logger.warning("토크나이저 저장 실패 (무시): %s", e)
