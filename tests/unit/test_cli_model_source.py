"""ModelSourcePlan CLI source resolution tests."""

from __future__ import annotations

import pytest
import typer

from mdp.cli.output import ModelSourcePlan, resolve_model_source_plan


class TestResolveModelSourcePlan:
    def test_model_dir_returns_artifact_plan(self) -> None:
        plan = resolve_model_source_plan(None, "/models/exported", "inference")

        assert isinstance(plan, ModelSourcePlan)
        assert plan.kind == "artifact"
        assert plan.command == "inference"
        assert str(plan.path) == "/models/exported"
        assert plan.uri is None
        assert plan.supports_pretrained is True

    def test_run_id_downloads_artifact_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mlflow.artifacts.download_artifacts",
            lambda run_id, artifact_path: f"/tmp/{run_id}/{artifact_path}",
        )

        plan = resolve_model_source_plan("abc123", None, "generate")

        assert plan.kind == "artifact"
        assert str(plan.path) == "/tmp/abc123/model"
        assert plan.uri is None

    def test_pretrained_returns_pretrained_plan(self) -> None:
        plan = resolve_model_source_plan(
            None, None, "generate", pretrained="hf://gpt2",
        )

        assert plan.kind == "pretrained"
        assert plan.path is None
        assert plan.uri == "hf://gpt2"
        assert plan.is_pretrained
        assert not plan.is_artifact

    def test_sources_are_mutually_exclusive(self) -> None:
        with pytest.raises(typer.BadParameter, match="하나만"):
            resolve_model_source_plan(
                "abc123", "/models/exported", "inference", pretrained="hf://gpt2",
            )

    def test_serve_does_not_support_direct_pretrained(self) -> None:
        with pytest.raises(typer.BadParameter, match="inference, generate"):
            resolve_model_source_plan(
                None, None, "serve", pretrained="hf://gpt2",
            )

    def test_serve_artifact_plan_marks_pretrained_unsupported(self) -> None:
        plan = resolve_model_source_plan(None, "/models/exported", "serve")

        assert plan.kind == "artifact"
        assert plan.supports_pretrained is False
