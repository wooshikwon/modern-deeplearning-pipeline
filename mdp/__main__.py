from dotenv import load_dotenv

load_dotenv()

import typer

from mdp.cli.output import OutputFormat, apply_format_override, set_output_format

app = typer.Typer(
    name="mdp",
    help=(
        "Modern Deeplearning Pipeline -- YAML 설정 기반 DL 학습/추론/서빙 CLI\n\n"
        "AI/LLM agents: Read AGENT.md for complete CLI reference, "
        "YAML schema, and working examples."
    ),
    no_args_is_help=True,
    invoke_without_command=True,
)


# 모든 subcommand에서 공유하는 --format 옵션. None이면 최상위 callback의 값을 유지.
# 명시되면 apply_format_override가 글로벌 포맷을 덮어쓴다.
# 이 패턴으로 `mdp --format json <cmd>`와 `mdp <cmd> --format json` 둘 다 작동한다.
_FORMAT_HELP = "출력 형식: text | json. 최상위 옵션과 동일하며 subcommand 위치에서도 사용 가능"



@app.callback()
def main(
    format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="출력 형식. text: 사람용 Rich 테이블, json: 기계 파싱용 JSON",
    ),
):
    """Modern Deeplearning Pipeline -- YAML 설정 기반 DL 학습/추론/서빙 CLI

    AI/LLM agents: Read AGENT.md for complete CLI reference,
    YAML schema, and working examples.
    """
    set_output_format(format)


@app.command()
def version(
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """MDP 버전을 출력한다."""
    apply_format_override(format)
    from mdp import __version__
    from mdp.cli.output import build_result, emit_result, is_json_mode

    if is_json_mode():
        result = build_result(command="version", version=__version__)
        emit_result(result)
    else:
        typer.echo(f"mdp {__version__}")


@app.command()
def train(
    recipe: str = typer.Option(..., "-r", "--recipe", help="Recipe YAML 경로"),
    config: str = typer.Option(..., "-c", "--config", help="Config YAML 경로"),
    callbacks: str = typer.Option(None, "--callbacks", help="콜백 YAML 파일 경로 (Recipe callbacks를 override)"),
    override: list[str] | None = typer.Option(None, "--override", help="오버라이드. KEY=VALUE 또는 JSON dict. 예: --override training.epochs=3 --override data.batch_size=8, 또는 --override '{\"training.epochs\": 3, \"data.batch_size\": 8}'"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """모델을 학습한다."""
    apply_format_override(format)
    from mdp.cli.train import run_train

    run_train(recipe, config, overrides=override, callbacks_file=callbacks)


@app.command(name="rl-train")
def rl_train(
    recipe: str = typer.Option(..., "-r", "--recipe", help="RL Recipe YAML 경로"),
    config: str = typer.Option(..., "-c", "--config", help="Config YAML 경로"),
    callbacks: str = typer.Option(None, "--callbacks", help="콜백 YAML 파일 경로 (Recipe callbacks를 override)"),
    override: list[str] | None = typer.Option(None, "--override", help="오버라이드. KEY=VALUE 또는 JSON dict. 예: --override training.epochs=3 --override data.batch_size=8, 또는 --override '{\"training.epochs\": 3, \"data.batch_size\": 8}'"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """RL alignment 학습. 내장 DPO/GRPO/PPO + `_component_`로 외부 알고리즘 주입."""
    apply_format_override(format)
    from mdp.cli.rl_train import run_rl_train

    run_rl_train(recipe, config, overrides=override, callbacks_file=callbacks)


@app.command()
def inference(
    run_id: str = typer.Option(None, "--run-id", help="MLflow run ID"),
    model_dir: str = typer.Option(None, "--model-dir", help="로컬 모델 디렉토리 (mdp export 결과)"),
    pretrained: str = typer.Option(None, "--pretrained", help="사전학습 모델 URI (hf://model-name)"),
    tokenizer: str = typer.Option(None, "--tokenizer", help="토크나이저 이름 (--pretrained 시 자동 추론, 명시 가능)"),
    data: str = typer.Option(..., "--data", help="추론 대상 데이터 (HF Hub 이름 또는 로컬 경로)"),
    fields: list[str] | None = typer.Option(None, "--fields", help="필드 매핑 오버라이드 (예: image=img label=class)"),
    metrics: list[str] | None = typer.Option(None, "--metrics", help="평가 metric (예: Accuracy F1Score)"),
    callbacks: str = typer.Option(None, "--callbacks", help="콜백 YAML 파일 경로 (추론 콜백)"),
    save_output: bool = typer.Option(False, "--save-output", help="콜백 전용 모드에서도 DefaultOutputCallback으로 결과 파일을 저장"),
    output_format: str = typer.Option("parquet", "--output-format", help="결과 포맷: parquet|csv|jsonl"),
    output_dir: str = typer.Option("./output", "--output-dir", help="결과 저장 디렉토리"),
    device_map: str = typer.Option(None, "--device-map", help="multi-GPU 분산 배치: auto|balanced|sequential"),
    dtype: str = typer.Option(None, "--dtype", help="모델 로딩 dtype: float32|float16|bfloat16"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code/--no-trust-remote-code", help="HF 모델의 remote code 신뢰 여부"),
    attn_impl: str = typer.Option(None, "--attn-impl", help="어텐션 구현: flash_attention_2|sdpa|eager"),
    batch_size: int = typer.Option(32, "--batch-size", help="pretrained 추론 배치 크기 (대형 모델은 줄여서 OOM 방지)"),
    max_length: int = typer.Option(512, "--max-length", help="토큰화 최대 길이 (pretrained 경로)"),
    override: list[str] | None = typer.Option(None, "--override", help="Recipe 오버라이드 (KEY=VALUE)"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """배치 추론을 실행한다. --run-id, --model-dir, --pretrained 중 하나를 지정."""
    apply_format_override(format)
    from mdp.cli.inference import run_inference

    run_inference(
        run_id, model_dir, data, fields, metrics, output_format, output_dir,
        device_map=device_map, overrides=override,
        pretrained=pretrained, tokenizer_name=tokenizer,
        callbacks_file=callbacks,
        dtype=dtype, trust_remote_code=trust_remote_code, attn_impl=attn_impl,
        save_output=save_output,
        batch_size=batch_size, max_length=max_length,
    )


@app.command()
def generate(
    run_id: str = typer.Option(None, "--run-id", help="MLflow run ID"),
    model_dir: str = typer.Option(None, "--model-dir", help="로컬 모델 디렉토리 (mdp export 결과)"),
    pretrained: str = typer.Option(None, "--pretrained", help="사전학습 모델 URI (hf://model-name)"),
    tokenizer: str = typer.Option(None, "--tokenizer", help="토크나이저 이름 (--pretrained 시 자동 추론, 명시 가능)"),
    prompts: str = typer.Option(..., "--prompts", help="프롬프트 JSONL 파일 경로"),
    prompt_field: str = typer.Option("prompt", "--prompt-field", help="JSONL에서 프롬프트 텍스트 필드명"),
    output: str = typer.Option("./generated.jsonl", "--output", "-o", help="출력 JSONL 경로"),
    max_new_tokens: int = typer.Option(None, "--max-new-tokens", help="생성 최대 토큰 수"),
    temperature: float = typer.Option(None, "--temperature", help="샘플링 temperature"),
    top_p: float = typer.Option(None, "--top-p", help="nucleus sampling p"),
    top_k: int = typer.Option(None, "--top-k", help="top-k sampling"),
    do_sample: bool = typer.Option(None, "--do-sample", help="샘플링 사용 여부"),
    num_samples: int = typer.Option(1, "--num-samples", help="프롬프트당 생성 횟수"),
    batch_size: int = typer.Option(1, "--batch-size", help="배치 크기"),
    callbacks: str = typer.Option(None, "--callbacks", help="콜백 YAML 파일 경로 (추론 콜백)"),
    device_map: str = typer.Option(None, "--device-map", help="multi-GPU 분산 배치: auto|balanced|sequential"),
    dtype: str = typer.Option(None, "--dtype", help="모델 로딩 dtype: float32|float16|bfloat16"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code/--no-trust-remote-code", help="HF 모델의 remote code 신뢰 여부"),
    attn_impl: str = typer.Option(None, "--attn-impl", help="어텐션 구현: flash_attention_2|sdpa|eager"),
    override: list[str] | None = typer.Option(None, "--override", help="Recipe 오버라이드 (KEY=VALUE)"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """프롬프트 JSONL에서 autoregressive 생성을 실행한다."""
    apply_format_override(format)
    from mdp.cli.generate import run_generate

    run_generate(
        run_id, model_dir, prompts, prompt_field, output,
        max_new_tokens, temperature, top_p, top_k, do_sample,
        num_samples, batch_size, device_map, overrides=override,
        pretrained=pretrained, tokenizer_name=tokenizer,
        callbacks_file=callbacks,
        dtype=dtype, trust_remote_code=trust_remote_code, attn_impl=attn_impl,
    )


@app.command()
def serve(
    run_id: str = typer.Option(None, "--run-id", help="MLflow run ID"),
    model_dir: str = typer.Option(None, "--model-dir", help="로컬 모델 디렉토리 (mdp export 결과)"),
    port: int = typer.Option(8000, "--port", help="서버 포트"),
    host: str = typer.Option("0.0.0.0", "--host", help="바인드 주소"),
    device_map: str = typer.Option(None, "--device-map", help="multi-GPU 분산 배치: auto|balanced|sequential"),
    max_memory: str = typer.Option(None, "--max-memory", help='GPU별 최대 메모리 (JSON): \'{"0":"24GiB","1":"40GiB"}\''),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """모델을 REST API로 서빙한다. --run-id 또는 --model-dir 중 하나를 지정."""
    apply_format_override(format)
    from mdp.cli.serve import run_serve

    run_serve(run_id, model_dir, port, host, device_map=device_map, max_memory=max_memory)


@app.command()
def export(
    run_id: str = typer.Option(None, "--run-id", help="MLflow run ID"),
    checkpoint: str = typer.Option(None, "--checkpoint", help="로컬 checkpoint 디렉토리"),
    output: str = typer.Option("./exported-model", "--output", "-o", help="출력 디렉토리"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """모델을 서빙 가능한 형태로 내보낸다 (adapter merge + 패키징)."""
    apply_format_override(format)
    from mdp.cli.export import run_export

    run_export(run_id, checkpoint, output)


@app.command()
def init(
    name: str = typer.Argument(..., help="프로젝트 이름"),
    task: str = typer.Option(None, "--task", "-t", help="task 이름 (catalog 기반 Recipe 생성)"),
    model: str = typer.Option(None, "--model", "-m", help="모델 이름 (catalog에서 조회)"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """새 MDP 프로젝트를 생성한다."""
    apply_format_override(format)
    from mdp.cli.init import init_project

    init_project(name, task=task, model=model)


@app.command()
def estimate(
    recipe: str = typer.Option(..., "-r", "--recipe", help="Recipe YAML 경로"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """GPU 메모리 사용량을 추정한다."""
    apply_format_override(format)
    from mdp.cli.estimate import run_estimate

    run_estimate(recipe)


@app.command(name="list")
def list_cmd(
    target: str = typer.Argument("models", help="models|tasks|callbacks|strategies"),
    task: str = typer.Option(None, "--task", "-t", help="models 필터: 특정 task 호환 모델만 표시"),
    format: OutputFormat | None = typer.Option(None, "--format", help=_FORMAT_HELP),
):
    """카탈로그를 조회한다."""
    apply_format_override(format)
    from mdp.cli.list_cmd import run_list

    run_list(target, task_filter=task)


if __name__ == "__main__":
    app()
