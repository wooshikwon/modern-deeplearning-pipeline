"""--pretrained CLI 유닛 테스트: resolve_model_source 3-way + tokenizer 추론."""

from __future__ import annotations

import pytest
import torch
import typer

from mdp.cli.output import resolve_model_source
from mdp.cli.generate import _resolve_pretrained_tokenizer_name
from mdp.models.pretrained import PretrainedLoadSpec


# ── resolve_model_source 3-way 상호 배타 ──


class TestResolveModelSource:
    def test_run_id_only(self, monkeypatch):
        """run_id만 지정 시 mlflow에서 Path를 반환한다."""
        monkeypatch.setattr(
            "mlflow.artifacts.download_artifacts",
            lambda run_id, artifact_path: f"/tmp/{run_id}/{artifact_path}",
        )
        result = resolve_model_source("abc123", None, "inference")
        assert result is not None
        assert "abc123" in str(result)

    def test_model_dir_only(self):
        """model_dir만 지정 시 Path를 반환한다."""
        result = resolve_model_source(None, "/some/path", "inference")
        assert result is not None
        assert str(result) == "/some/path"

    def test_pretrained_only(self):
        """pretrained만 지정 시 None을 반환한다."""
        result = resolve_model_source(None, None, "inference", pretrained="hf://gpt2")
        assert result is None

    def test_pretrained_returns_none_for_generate(self):
        """generate 커맨드에서도 pretrained는 None을 반환한다."""
        result = resolve_model_source(None, None, "generate", pretrained="hf://gpt2")
        assert result is None

    def test_two_sources_raises(self):
        """run_id + model_dir 동시 지정 시 에러."""
        with pytest.raises(typer.BadParameter, match="하나만"):
            resolve_model_source("abc", "/path", "inference")

    def test_run_id_and_pretrained_raises(self):
        """run_id + pretrained 동시 지정 시 에러."""
        with pytest.raises(typer.BadParameter, match="하나만"):
            resolve_model_source("abc", None, "inference", pretrained="hf://gpt2")

    def test_model_dir_and_pretrained_raises(self):
        """model_dir + pretrained 동시 지정 시 에러."""
        with pytest.raises(typer.BadParameter, match="하나만"):
            resolve_model_source(None, "/path", "inference", pretrained="hf://gpt2")

    def test_all_three_raises(self):
        """3개 모두 지정 시 에러."""
        with pytest.raises(typer.BadParameter, match="하나만"):
            resolve_model_source("abc", "/path", "inference", pretrained="hf://gpt2")

    def test_none_raises(self):
        """하나도 지정하지 않으면 에러."""
        with pytest.raises(typer.BadParameter, match="하나가 필요"):
            resolve_model_source(None, None, "inference")

    def test_pretrained_serve_raises(self):
        """serve 커맨드에서 pretrained 사용 시 에러."""
        with pytest.raises(typer.BadParameter, match="inference, generate"):
            resolve_model_source(None, None, "serve", pretrained="hf://gpt2")

    def test_pretrained_export_raises(self):
        """export 커맨드에서 pretrained 사용 시 에러."""
        with pytest.raises(typer.BadParameter, match="inference, generate"):
            resolve_model_source(None, None, "export", pretrained="hf://gpt2")

    def test_backward_compat_no_pretrained(self):
        """pretrained 인자 없이 호출해도 기존대로 동작한다."""
        result = resolve_model_source(None, "/old/path", "serve")
        assert str(result) == "/old/path"


# ── _resolve_pretrained_tokenizer_name ──


class TestResolvePretrainedTokenizerName:
    def test_hf_prefix(self):
        """hf:// 접두사가 제거된다."""
        assert _resolve_pretrained_tokenizer_name("hf://meta-llama/Meta-Llama-3-8B") == "meta-llama/Meta-Llama-3-8B"

    def test_no_prefix(self):
        """접두사 없으면 그대로 반환한다."""
        assert _resolve_pretrained_tokenizer_name("gpt2") == "gpt2"

    def test_timm_raises(self):
        """timm:// 모델은 토크나이저 자동 추론 불가."""
        with pytest.raises(ValueError, match="timm://"):
            _resolve_pretrained_tokenizer_name("timm://resnet50")

    def test_ultralytics_raises(self):
        """ultralytics:// 모델은 토크나이저 자동 추론 불가."""
        with pytest.raises(ValueError, match="ultralytics://"):
            _resolve_pretrained_tokenizer_name("ultralytics://yolov8n.pt")

    def test_local_raises(self):
        """local:// 모델은 토크나이저 자동 추론 불가."""
        with pytest.raises(ValueError, match="local://"):
            _resolve_pretrained_tokenizer_name("local:///path/to/model.pt")


# ── PretrainedLoadSpec option normalization ──


class TestPretrainedLoadSpec:
    def test_hf_cli_options_are_normalized(self):
        spec = PretrainedLoadSpec.from_options(
            "hf://gpt2",
            dtype="bfloat16",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        assert spec.protocol == "hf"
        assert spec.torch_dtype is torch.bfloat16
        assert spec.to_loader_kwargs() == {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
        }

    def test_no_prefix_defaults_to_hf_protocol(self):
        spec = PretrainedLoadSpec.from_options("gpt2")

        assert spec.protocol == "hf"
        assert spec.uri == "gpt2"
        assert spec.to_loader_kwargs() == {}

    def test_invalid_dtype_raises_clear_error(self):
        with pytest.raises(ValueError, match="지원하지 않는 torch dtype"):
            PretrainedLoadSpec.from_options("hf://gpt2", dtype="not_a_dtype")

    def test_non_hf_rejects_hf_only_options(self):
        with pytest.raises(ValueError, match="device_map"):
            PretrainedLoadSpec.from_options("timm://resnet50", device_map="auto")

    def test_local_rejects_dtype(self):
        with pytest.raises(ValueError, match="torch_dtype"):
            PretrainedLoadSpec.from_options("local:///tmp/model.pt", dtype="float16")
