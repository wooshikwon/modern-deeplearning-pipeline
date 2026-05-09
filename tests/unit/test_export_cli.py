"""mdp export unit tests."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from mdp.settings.components import ComponentSpec


def test_export_saves_tokenizer_from_typed_collator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Export reads tokenizer from ComponentSpec.kwargs, not raw YAML dicts."""
    from mdp.cli.export import run_export
    from mdp.serving import model_loader

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "exported"
    source_dir.mkdir()
    (source_dir / "recipe.yaml").write_text("name: export-test\n")

    class _Tokenizer:
        def save_pretrained(self, output_dir: Path) -> None:
            (Path(output_dir) / "tokenizer.json").write_text("{}")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str) -> _Tokenizer:
            assert name == "gpt2"
            return _Tokenizer()

    settings = SimpleNamespace(
        recipe=SimpleNamespace(
            data=SimpleNamespace(
                collator=ComponentSpec(
                    component="mdp.data.collators.CausalLMCollator",
                    kwargs={"tokenizer": "gpt2"},
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        model_loader,
        "reconstruct_model",
        lambda source, merge: (torch.nn.Linear(2, 1), settings),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )

    run_export(checkpoint=str(source_dir), output=str(output_dir))

    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "recipe.yaml").exists()
