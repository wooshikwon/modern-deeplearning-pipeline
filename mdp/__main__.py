import typer

from mdp.cli.output import OutputFormat, set_output_format

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
def version():
    """MDP 버전을 출력한다."""
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
):
    """모델을 학습한다."""
    from mdp.cli.train import run_train

    run_train(recipe, config)


@app.command(name="rl-train")
def rl_train(
    recipe: str = typer.Option(..., "-r", "--recipe", help="RL Recipe YAML 경로"),
    config: str = typer.Option(..., "-c", "--config", help="Config YAML 경로"),
):
    """RL alignment 학습 (DPO, weighted-NTP, GRPO, PPO)."""
    from mdp.cli.rl_train import run_rl_train

    run_rl_train(recipe, config)


@app.command()
def inference(
    run_id: str = typer.Option(None, "--run-id", help="MLflow run ID"),
    model_dir: str = typer.Option(None, "--model-dir", help="로컬 모델 디렉토리 (mdp export 결과)"),
    data: str = typer.Option(..., "--data", help="추론 대상 데이터 (HF Hub 이름 또는 로컬 경로)"),
    fields: list[str] | None = typer.Option(None, "--fields", help="필드 매핑 오버라이드 (예: image=img label=class)"),
    metrics: list[str] | None = typer.Option(None, "--metrics", help="평가 metric (예: Accuracy F1Score)"),
    output_format: str = typer.Option("parquet", "--output-format", help="결과 포맷: parquet|csv|jsonl"),
    output_dir: str = typer.Option("./output", "--output-dir", help="결과 저장 디렉토리"),
):
    """배치 추론을 실행한다. --run-id 또는 --model-dir 중 하나를 지정."""
    from mdp.cli.inference import run_inference

    run_inference(run_id, model_dir, data, fields, metrics, output_format, output_dir)


@app.command()
def serve(
    run_id: str = typer.Option(None, "--run-id", help="MLflow run ID"),
    model_dir: str = typer.Option(None, "--model-dir", help="로컬 모델 디렉토리 (mdp export 결과)"),
    port: int = typer.Option(8000, "--port", help="서버 포트"),
    host: str = typer.Option("0.0.0.0", "--host", help="바인드 주소"),
):
    """모델을 REST API로 서빙한다. --run-id 또는 --model-dir 중 하나를 지정."""
    from mdp.cli.serve import run_serve

    run_serve(run_id, model_dir, port, host)


@app.command()
def export(
    run_id: str = typer.Option(None, "--run-id", help="MLflow run ID"),
    checkpoint: str = typer.Option(None, "--checkpoint", help="로컬 checkpoint 디렉토리"),
    output: str = typer.Option("./exported-model", "--output", "-o", help="출력 디렉토리"),
):
    """모델을 서빙 가능한 형태로 내보낸다 (adapter merge + 패키징)."""
    from mdp.cli.export import run_export

    run_export(run_id, checkpoint, output)


@app.command()
def init(
    name: str = typer.Argument(..., help="프로젝트 이름"),
    task: str = typer.Option(None, "--task", "-t", help="task 이름 (catalog 기반 Recipe 생성)"),
    model: str = typer.Option(None, "--model", "-m", help="모델 이름 (catalog에서 조회)"),
):
    """새 MDP 프로젝트를 생성한다."""
    from mdp.cli.init import init_project

    init_project(name, task=task, model=model)


@app.command()
def estimate(
    recipe: str = typer.Option(..., "-r", "--recipe", help="Recipe YAML 경로"),
):
    """GPU 메모리 사용량을 추정한다."""
    from mdp.cli.estimate import run_estimate

    run_estimate(recipe)


@app.command(name="list")
def list_cmd(
    target: str = typer.Argument("models", help="models|tasks|callbacks|strategies"),
    task: str = typer.Option(None, "--task", "-t", help="models 필터: 특정 task 호환 모델만 표시"),
):
    """카탈로그를 조회한다."""
    from mdp.cli.list_cmd import run_list

    run_list(target, task_filter=task)


if __name__ == "__main__":
    app()
