"""config.architectures 기반 모델 클래스 해석 + PretrainedResolver.load 유닛 테스트."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from mdp.models.pretrained import PretrainedResolver


def _make_mock_transformers(**extra_attrs):
    """transformers 모듈의 mock을 생성한다.

    AutoConfig를 기본 제공하고, extra_attrs로 추가 클래스를 설정한다.
    """
    mock_tf = MagicMock()
    for name, val in extra_attrs.items():
        setattr(mock_tf, name, val)
    return mock_tf


# ── _load_hf: config.architectures 직접 클래스 해석 ──


class TestLoadHfByArchitectures:
    """class_path=None일 때 config.architectures[0]에서 직접 모델 클래스를 결정하는지 검증."""

    @pytest.mark.parametrize("arch_name", [
        "GPT2LMHeadModel",
        "LlamaForCausalLM",
        "BertForSequenceClassification",
        "ViTForImageClassification",
    ])
    def test_architectures_resolves_to_class(self, arch_name):
        """config.architectures[0]의 클래스명으로 transformers에서 직접 클래스를 가져온다."""
        mock_cls = MagicMock(__name__=arch_name)
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=[arch_name])
        mock_tf = _make_mock_transformers(**{arch_name: mock_cls})
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            result = PretrainedResolver._load_hf("some-model")
        mock_cls.from_pretrained.assert_called_once_with("some-model")
        assert result is mock_cls.from_pretrained.return_value

    def test_no_architectures_raises_error(self):
        """config.architectures가 없으면 ValueError를 발생시킨다."""
        mock_config = SimpleNamespace(architectures=None)
        mock_tf = _make_mock_transformers()
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            with pytest.raises(ValueError, match="architectures 필드가 없습니다"):
                PretrainedResolver._load_hf("some-model")

    def test_empty_architectures_raises_error(self):
        """config.architectures가 빈 리스트이면 ValueError를 발생시킨다."""
        mock_config = SimpleNamespace(architectures=[])
        mock_tf = _make_mock_transformers()
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            with pytest.raises(ValueError, match="architectures 필드가 없습니다"):
                PretrainedResolver._load_hf("some-model")

    def test_unknown_class_raises_error(self):
        """transformers에 해당 클래스가 없으면 ValueError를 발생시킨다."""
        mock_config = SimpleNamespace(architectures=["NonExistentModel"])
        # spec=[]로 생성하면 정의되지 않은 속성 접근 시 AttributeError가 발생한다.
        # getattr(mock_tf, "NonExistentModel", None)이 None을 반환하게 된다.
        mock_tf = MagicMock(spec=[])
        mock_tf.AutoConfig = MagicMock()
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            with pytest.raises(ValueError, match="'NonExistentModel' 클래스가 없습니다"):
                PretrainedResolver._load_hf("some-model")

    def test_first_architecture_is_used(self):
        """architectures 리스트의 첫 번째 항목이 사용된다."""
        first_cls = MagicMock(__name__="FirstModel")
        first_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=["FirstModel", "SecondModel"])
        mock_tf = _make_mock_transformers(FirstModel=first_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            result = PretrainedResolver._load_hf("some-model")
        first_cls.from_pretrained.assert_called_once_with("some-model")
        assert result is first_cls.from_pretrained.return_value


# ── PretrainedResolver.load: 파라미터 전달 ──


class TestPretrainedResolverLoad:
    """PretrainedResolver.load()가 올바른 파라미터를 전달하는지 검증."""

    def test_hf_without_class_path(self):
        """class_path 없이 호출하면 _load_hf에 class_path=None이 전달된다."""
        mock_model = MagicMock()
        with patch.object(PretrainedResolver, "_load_hf", return_value=mock_model) as mock_load:
            PretrainedResolver.load("hf://gpt2")
        mock_load.assert_called_once_with("gpt2", class_path=None)

    def test_hf_with_class_path(self):
        """class_path가 명시되면 그대로 전달된다."""
        mock_model = MagicMock()
        with patch.object(PretrainedResolver, "_load_hf", return_value=mock_model) as mock_load:
            PretrainedResolver.load(
                "hf://gpt2",
                class_path="transformers.AutoModelForCausalLM",
            )
        mock_load.assert_called_once_with(
            "gpt2",
            class_path="transformers.AutoModelForCausalLM",
        )

    def test_non_hf_protocol_no_class_path(self):
        """hf:// 이외 프로토콜은 class_path를 전달하지 않는다."""
        mock_model = MagicMock()
        with patch.object(PretrainedResolver, "_load_timm", return_value=mock_model) as mock_load:
            PretrainedResolver.load("timm://resnet50")
        mock_load.assert_called_once_with("resnet50")

    def test_no_prefix_defaults_to_hf(self):
        """접두사 없는 URI는 hf://로 취급된다."""
        mock_model = MagicMock()
        with patch.object(PretrainedResolver, "_load_hf", return_value=mock_model) as mock_load:
            PretrainedResolver.load("bert-base-uncased", class_path="transformers.AutoModel")
        mock_load.assert_called_once_with(
            "bert-base-uncased",
            class_path="transformers.AutoModel",
        )


# ── _load_hf: class_path 분기 ──


class TestLoadHfClassPathBranch:
    """_load_hf가 class_path 유무에 따라 올바른 분기를 타는지 검증."""

    def test_class_path_skips_config_lookup(self):
        """class_path가 있으면 config.architectures 조회를 하지 않는다."""
        mock_model = MagicMock()
        mock_tf = _make_mock_transformers()
        with (
            patch.dict(sys.modules, {"transformers": mock_tf}),
            patch("importlib.import_module") as mock_import,
        ):
            mock_module = MagicMock()
            mock_cls = MagicMock(__name__="AutoModelForCausalLM")
            mock_cls.from_pretrained.return_value = mock_model
            mock_module.AutoModelForCausalLM = mock_cls
            mock_import.return_value = mock_module
            PretrainedResolver._load_hf(
                "gpt2", class_path="transformers.AutoModelForCausalLM",
            )
        mock_tf.AutoConfig.from_pretrained.assert_not_called()

    def test_no_class_path_uses_config_architectures(self):
        """class_path가 None이면 config.architectures에서 클래스를 결정한다."""
        mock_cls = MagicMock(__name__="GPT2LMHeadModel")
        mock_model = MagicMock()
        mock_cls.from_pretrained.return_value = mock_model
        mock_config = SimpleNamespace(architectures=["GPT2LMHeadModel"])
        mock_tf = _make_mock_transformers(GPT2LMHeadModel=mock_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            result = PretrainedResolver._load_hf("gpt2")
        mock_tf.AutoConfig.from_pretrained.assert_called_once_with("gpt2")
        assert result is mock_model


# ── kwargs 전달: dtype, trust_remote_code, device_map, attn_implementation ──


class TestPretrainedKwargsForwarding:
    """CLI kwargs가 PretrainedResolver.load → from_pretrained까지 도달하는지 검증."""

    def test_dtype_forwarded_to_from_pretrained(self):
        """torch_dtype kwarg이 from_pretrained에 전달된다."""
        mock_cls = MagicMock(__name__="GPT2LMHeadModel")
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=["GPT2LMHeadModel"])
        mock_tf = _make_mock_transformers(GPT2LMHeadModel=mock_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            PretrainedResolver.load("hf://some-model", torch_dtype=torch.float16)
        mock_cls.from_pretrained.assert_called_once_with(
            "some-model", torch_dtype=torch.float16,
        )

    def test_trust_remote_code_forwarded(self):
        """trust_remote_code kwarg이 from_pretrained에 전달된다."""
        mock_cls = MagicMock(__name__="Qwen2ForCausalLM")
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=["Qwen2ForCausalLM"])
        mock_tf = _make_mock_transformers(Qwen2ForCausalLM=mock_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            PretrainedResolver.load("hf://qwen-model", trust_remote_code=True)
        mock_cls.from_pretrained.assert_called_once_with(
            "qwen-model", trust_remote_code=True,
        )

    def test_device_map_forwarded(self):
        """device_map kwarg이 from_pretrained에 전달된다."""
        mock_cls = MagicMock(__name__="LlamaForCausalLM")
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        mock_tf = _make_mock_transformers(LlamaForCausalLM=mock_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            PretrainedResolver.load("hf://llama-model", device_map="auto")
        mock_cls.from_pretrained.assert_called_once_with(
            "llama-model", device_map="auto",
        )

    def test_attn_implementation_forwarded(self):
        """attn_implementation kwarg이 from_pretrained에 전달된다."""
        mock_cls = MagicMock(__name__="GPT2LMHeadModel")
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=["GPT2LMHeadModel"])
        mock_tf = _make_mock_transformers(GPT2LMHeadModel=mock_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            PretrainedResolver.load("hf://gpt2", attn_implementation="sdpa")
        mock_cls.from_pretrained.assert_called_once_with(
            "gpt2", attn_implementation="sdpa",
        )

    def test_multiple_kwargs_combined(self):
        """여러 kwargs가 동시에 from_pretrained에 전달된다."""
        mock_cls = MagicMock(__name__="GPT2LMHeadModel")
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=["GPT2LMHeadModel"])
        mock_tf = _make_mock_transformers(GPT2LMHeadModel=mock_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            PretrainedResolver.load(
                "hf://gpt2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        mock_cls.from_pretrained.assert_called_once_with(
            "gpt2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

    def test_no_kwargs_clean_call(self):
        """kwargs가 없으면 from_pretrained에 추가 인자가 전달되지 않는다."""
        mock_cls = MagicMock(__name__="GPT2LMHeadModel")
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_config = SimpleNamespace(architectures=["GPT2LMHeadModel"])
        mock_tf = _make_mock_transformers(GPT2LMHeadModel=mock_cls)
        mock_tf.AutoConfig.from_pretrained.return_value = mock_config
        with patch.dict(sys.modules, {"transformers": mock_tf}):
            PretrainedResolver.load("hf://gpt2")
        mock_cls.from_pretrained.assert_called_once_with("gpt2")


# ── _build_pretrained_kwargs 헬퍼 ──


class TestBuildPretrainedKwargs:
    """_build_pretrained_kwargs가 올바른 dict를 구성하는지 검증."""

    def test_all_none(self):
        """모든 인자가 None/False이면 빈 dict를 반환한다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs(None, False, None, None)
        assert result == {}

    def test_dtype_converts_to_torch_dtype(self):
        """dtype 문자열이 torch dtype으로 변환된다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs("float16", False, None, None)
        assert result == {"torch_dtype": torch.float16}

    def test_bfloat16_dtype(self):
        """bfloat16 문자열이 torch.bfloat16으로 변환된다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs("bfloat16", False, None, None)
        assert result == {"torch_dtype": torch.bfloat16}

    def test_trust_remote_code_true(self):
        """trust_remote_code=True가 포함된다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs(None, True, None, None)
        assert result == {"trust_remote_code": True}

    def test_trust_remote_code_false_omitted(self):
        """trust_remote_code=False면 dict에 포함되지 않는다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs(None, False, None, None)
        assert "trust_remote_code" not in result

    def test_attn_impl(self):
        """attn_impl이 attn_implementation으로 전달된다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs(None, False, "sdpa", None)
        assert result == {"attn_implementation": "sdpa"}

    def test_device_map(self):
        """device_map이 포함된다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs(None, False, None, "auto")
        assert result == {"device_map": "auto"}

    def test_all_set(self):
        """모든 인자가 설정되면 전부 포함된다."""
        from mdp.cli.inference import _build_pretrained_kwargs

        result = _build_pretrained_kwargs("float32", True, "flash_attention_2", "balanced")
        assert result == {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
            "device_map": "balanced",
        }
