"""mdp generate -- autoregressive 텍스트 생성."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode, resolve_model_source

logger = logging.getLogger(__name__)


def _resolve_tokenizer_name(settings: Any) -> str:
    """Recipe에서 토크나이저 이름을 추출한다.

    탐색 순서: collator.tokenizer → dataset.tokenizer → model.pretrained.
    """
    recipe = settings.recipe

    # 1. collator의 tokenizer 필드
    tok = recipe.data.collator.get("tokenizer")
    if tok:
        return tok

    # 2. dataset의 tokenizer 필드
    tok = recipe.data.dataset.get("tokenizer")
    if tok:
        return tok

    # 3. model의 pretrained 필드 (hf:// 접두사 제거)
    pretrained = recipe.model.get("pretrained", "")
    if pretrained.startswith("hf://"):
        return pretrained[5:]
    if pretrained:
        return pretrained

    raise ValueError(
        "토크나이저를 결정할 수 없습니다. "
        "recipe의 data.collator.tokenizer, data.dataset.tokenizer, "
        "또는 model.pretrained 중 하나가 필요합니다."
    )


def _resolve_pretrained_tokenizer_name(pretrained_uri: str) -> str:
    """pretrained URI에서 토크나이저 이름을 추론한다.

    hf:// 접두사가 있으면 제거하고 HuggingFace 모델 이름을 반환한다.
    접두사가 없으면 HF 모델명으로 간주한다.
    """
    if pretrained_uri.startswith("hf://"):
        return pretrained_uri[5:]
    for prefix in ("timm://", "ultralytics://", "local://"):
        if pretrained_uri.startswith(prefix):
            raise ValueError(
                f"{prefix} 모델의 토크나이저는 자동 추론할 수 없습니다. "
                "--tokenizer로 명시해 주세요."
            )
    return pretrained_uri


def run_generate(
    run_id: str | None,
    model_dir: str | None,
    prompts: str,
    prompt_field: str = "prompt",
    output: str = "./generated.jsonl",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    do_sample: bool | None = None,
    num_samples: int = 1,
    batch_size: int = 1,
    device_map: str | None = None,
    overrides: list[str] | None = None,
    pretrained: str | None = None,
    tokenizer_name: str | None = None,
    callbacks_file: str | None = None,
) -> None:
    """프롬프트 JSONL에서 autoregressive 생성을 실행한다."""
    import torch
    from transformers import AutoTokenizer

    try:
        model_path = resolve_model_source(run_id, model_dir, "generate", pretrained=pretrained)
    except typer.BadParameter as e:
        msg = str(e)
        if is_json_mode():
            emit_result(build_error(command="generate", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    try:
        is_pretrained = model_path is None

        if is_pretrained:
            # pretrained 분기: PretrainedResolver로 직접 로드, Recipe 없음
            from mdp.models.pretrained import PretrainedResolver

            model = PretrainedResolver.load(pretrained)
            model.eval()

            tok_name = tokenizer_name or _resolve_pretrained_tokenizer_name(pretrained)
            tokenizer = AutoTokenizer.from_pretrained(tok_name)
        else:
            # 기존 artifact 분기: reconstruct_model로 Recipe 기반 로드
            from mdp.serving.model_loader import reconstruct_model

            model, settings = reconstruct_model(
                model_path, merge=True, device_map=device_map, overrides=overrides,
            )
            model.eval()

            tok_name = tokenizer_name or _resolve_tokenizer_name(settings)
            tokenizer = AutoTokenizer.from_pretrained(tok_name)

        # 콜백 로드
        loaded_callbacks: list = []
        if callbacks_file:
            from mdp.settings.resolver import ComponentResolver
            from mdp.training._common import create_callbacks, load_callbacks_from_file

            cb_configs = load_callbacks_from_file(callbacks_file)
            loaded_callbacks = create_callbacks(cb_configs, ComponentResolver())
            if not is_json_mode():
                typer.echo(f"Callbacks: {len(loaded_callbacks)}개 로드 ({callbacks_file})")

        if not hasattr(model, "generate"):
            raise ValueError(
                f"모델 '{type(model).__name__}'에 generate 메서드가 없습니다. "
                "text_generation 태스크의 모델만 지원됩니다."
            )

        # Generation 파라미터: CLI defaults → recipe (artifact만) → CLI overrides
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "do_sample": True,
            "num_beams": 1,
            "repetition_penalty": 1.0,
        }
        if not is_pretrained:
            gen_spec = settings.recipe.generation
            if gen_spec is not None:
                gen_kwargs.update({
                    "max_new_tokens": gen_spec.max_new_tokens,
                    "temperature": gen_spec.temperature,
                    "top_p": gen_spec.top_p,
                    "top_k": gen_spec.top_k,
                    "do_sample": gen_spec.do_sample,
                    "num_beams": gen_spec.num_beams,
                    "repetition_penalty": gen_spec.repetition_penalty,
                })
        # CLI 값이 명시적으로 전달되었으면 override
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if do_sample is not None:
            gen_kwargs["do_sample"] = do_sample

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not is_json_mode():
            if is_pretrained:
                typer.echo(f"Pretrained: {pretrained}")
            elif run_id:
                typer.echo(f"MLflow run: {run_id}")
            else:
                typer.echo(f"모델 디렉토리: {model_dir}")
            typer.echo(f"토크나이저: {tok_name}")
            typer.echo(f"프롬프트: {prompts}")

        # JSONL 프롬프트 읽기
        prompt_records: list[dict[str, Any]] = []
        with open(prompts) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompt_records.append(json.loads(line))

        if not prompt_records:
            raise ValueError(f"프롬프트 파일이 비어 있습니다: {prompts}")

        if not is_json_mode():
            typer.echo(f"프롬프트 {len(prompt_records)}개 로드. 생성 시작...")

        # 배치 생성
        device = next(model.parameters()).device
        results: list[dict[str, Any]] = []

        for i in range(0, len(prompt_records), batch_size):
            batch_records = prompt_records[i:i + batch_size]
            batch_prompts = [r[prompt_field] for r in batch_records]

            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(device)

            for _ in range(num_samples):
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        pad_token_id=tokenizer.pad_token_id,
                        **gen_kwargs,
                    )

                # input 부분을 제외한 생성 텍스트만 디코딩
                input_length = inputs["input_ids"].shape[1]
                generated_ids = output_ids[:, input_length:]
                generated_texts = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True,
                )

                for record, text in zip(batch_records, generated_texts):
                    result = dict(record)
                    result["generated_text"] = text
                    results.append(result)

        # 결과 저장
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if not is_json_mode():
            typer.echo(f"생성 완료. {len(results)}건 → {output_path}")

        if is_json_mode():
            from mdp.cli.schemas import GenerateResult

            gen_result = GenerateResult(
                output_path=str(output_path),
                num_prompts=len(prompt_records),
                num_generated=len(results),
                num_samples=num_samples,
            )
            emit_result(build_result(
                command="generate", **gen_result.model_dump(exclude_none=True),
            ))

    except typer.Exit:
        raise
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="generate", error_type="RuntimeError", message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        logger.exception("생성 실패 상세")
        raise typer.Exit(code=1)
