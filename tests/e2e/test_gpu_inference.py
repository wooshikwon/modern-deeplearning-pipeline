"""GPU inference / generation e2e — forward pass and short autoregressive generate.

Validates: AutoModelForCausalLM forward + tokenizer round-trip on CUDA,
plus tiny generation (max_new_tokens=4) — exercises the inference path
without needing the full mdp inference CLI.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.fixtures
def test_causal_lm_forward_cuda(smollm2):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(smollm2))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(smollm2)).cuda().eval()

    inputs = tok(["hello world", "test prompt two"], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model(**inputs)

    assert out.logits.dim() == 3
    assert out.logits.shape[0] == 2
    assert torch.isfinite(out.logits).all().item()


@pytest.mark.gpu
@pytest.mark.fixtures
def test_causal_lm_generate_cuda(smollm2):
    """Short autoregressive generate on CUDA — exercises kv cache + sampling."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(smollm2))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(smollm2)).cuda().eval()

    prompt = tok("the cat", return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(
            **prompt,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    assert gen.shape[1] >= prompt.input_ids.shape[1] + 1
    decoded = tok.decode(gen[0], skip_special_tokens=True)
    assert isinstance(decoded, str)


@pytest.mark.gpu
@pytest.mark.fixtures
def test_seq_classification_forward_cuda(bert_tiny):
    """BERT classifier forward on CUDA — exercises encoder + classification head."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(bert_tiny))
    model = AutoModelForSequenceClassification.from_pretrained(str(bert_tiny)).cuda().eval()

    inputs = tok(["positive sentence", "another sentence"], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model(**inputs)

    assert out.logits.shape == (2, model.config.num_labels)
    assert torch.isfinite(out.logits).all().item()


@pytest.mark.gpu
@pytest.mark.fixtures
def test_vit_forward_cuda(vit_tiny):
    """ViT forward on CUDA — exercises vision branch."""
    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(str(vit_tiny)).cuda().eval()
    image_size = model.config.image_size

    pixel_values = torch.randn(2, 3, image_size, image_size, device="cuda")
    with torch.no_grad():
        out = model(pixel_values=pixel_values)

    assert out.logits.shape == (2, model.config.num_labels)
    assert torch.isfinite(out.logits).all().item()
