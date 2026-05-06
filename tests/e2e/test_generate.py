"""Generate runtime resolver integration tests."""

from __future__ import annotations

from types import SimpleNamespace


def test_resolve_generation_kwargs_precedence() -> None:
    from mdp.serving.handlers import resolve_generation_kwargs
    from mdp.settings.schema import GenerationSpec

    recipe_generation = GenerationSpec(
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        do_sample=False,
        num_beams=2,
        repetition_penalty=1.2,
    )

    resolved = resolve_generation_kwargs(
        recipe_generation,
        {"temperature": 0.2, "top_p": None, "do_sample": True},
    )

    assert resolved == {
        "max_new_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "do_sample": True,
        "num_beams": 2,
        "repetition_penalty": 1.2,
    }


def test_resolve_generation_kwargs_accepts_serving_arg_object() -> None:
    from mdp.serving.handlers import resolve_generation_kwargs

    resolved = resolve_generation_kwargs(
        {"max_new_tokens": 16, "temperature": 0.8},
        SimpleNamespace(max_new_tokens=4, top_k=10),
    )

    assert resolved["max_new_tokens"] == 4
    assert resolved["temperature"] == 0.8
    assert resolved["top_k"] == 10
    assert resolved["top_p"] == 1.0
