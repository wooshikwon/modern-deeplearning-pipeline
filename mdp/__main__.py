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


@app.command()
def inference(
    recipe: str = typer.Option(..., "-r", "--recipe", help="Recipe YAML 경로"),
    config: str = typer.Option(..., "-c", "--config", help="Config YAML 경로"),
    checkpoint: str = typer.Option(None, "--checkpoint", help="체크포인트 경로"),
):
    """배치 추론을 실행한다."""
    from mdp.cli.inference import run_inference

    run_inference(recipe, config, checkpoint)


@app.command()
def serve(
    recipe: str = typer.Option(..., "-r", "--recipe", help="Recipe YAML 경로"),
    config: str = typer.Option(..., "-c", "--config", help="Config YAML 경로"),
):
    """모델 서빙을 시작한다."""
    from mdp.cli.serve import run_serve

    run_serve(recipe, config)


@app.command()
def init(name: str = typer.Argument(..., help="프로젝트 이름")):
    """새 MDP 프로젝트를 생성한다."""
    from mdp.cli.init import init_project

    init_project(name)


@app.command()
def estimate(
    recipe: str = typer.Option(..., "-r", "--recipe", help="Recipe YAML 경로"),
):
    """GPU 메모리 사용량을 추정한다."""
    from mdp.cli.estimate import run_estimate

    run_estimate(recipe)


@app.command(name="list")
def list_cmd(target: str = typer.Argument("models", help="models|tasks|callbacks|strategies|jobs")):
    """카탈로그를 조회한다."""
    from mdp.cli.list_cmd import run_list

    run_list(target)


if __name__ == "__main__":
    app()
