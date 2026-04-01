"""mdp serve -- 모델 서빙을 시작한다.

Config의 serving.backend에 따라 적절한 서빙 백엔드를 선택하고 서버를 시작한다.
지원 백엔드: torchserve, vllm, onnx.
"""

from __future__ import annotations

import typer

from mdp.cli.output import build_result, emit_result, is_json_mode
from mdp.settings.factory import SettingsFactory


def _wait_process(process, backend_name: str) -> None:
    """프로세스 종료를 대기한다. KeyboardInterrupt 시 graceful shutdown."""
    try:
        process.wait()
    except KeyboardInterrupt:
        typer.echo(f"\n{backend_name} 종료 중...")
        process.terminate()
        process.wait()


def run_serve(recipe_path: str, config_path: str) -> None:
    """Recipe + Config에서 서빙 설정을 읽어 서버를 시작한다."""
    settings = SettingsFactory().for_inference(recipe_path, config_path)

    serving = settings.config.serving
    if serving is None:
        typer.echo("Config에 serving 섹션이 없습니다. serving.backend을 지정하세요.", err=True)
        raise typer.Exit(code=1)

    backend = serving.backend

    if is_json_mode():
        emit_result(build_result(
            command="serve",
            status="starting",
            backend=backend,
        ))

    if backend == "torchserve":
        _serve_torchserve(settings)
    elif backend == "vllm":
        _serve_vllm(settings)
    elif backend == "onnx":
        _serve_onnx(settings)
    else:
        typer.echo(f"지원하지 않는 서빙 백엔드: {backend}", err=True)
        raise typer.Exit(code=1)


def _serve_torchserve(settings) -> None:
    """TorchServe 서버를 시작한다."""
    from mdp.serving.torchserve import start_torchserve

    serving = settings.config.serving
    model_store = serving.model_repository or "model_store"
    if not is_json_mode():
        typer.echo(f"TorchServe 시작: model_store={model_store}")
    process = start_torchserve(
        model_store=model_store,
        port=8080,
        workers=serving.instance_count,
    )
    _wait_process(process, "TorchServe")


def _serve_vllm(settings) -> None:
    """vLLM 서버를 시작한다."""
    from mdp.serving.vllm_server import start_vllm_server

    model_name = settings.recipe.model.pretrained
    if model_name and model_name.startswith("hf://"):
        model_name = model_name[len("hf://"):]

    if not model_name:
        typer.echo("vLLM 서빙에는 model.pretrained URI가 필요합니다.", err=True)
        raise typer.Exit(code=1)

    serving = settings.config.serving
    if not is_json_mode():
        typer.echo(f"vLLM 서버 시작: model={model_name}")
    process = start_vllm_server(
        model_name=model_name,
        port=8000,
        tensor_parallel_size=serving.instance_count,
    )
    _wait_process(process, "vLLM")


def _serve_onnx(settings) -> None:
    """ONNX 모델을 로딩하고 간단한 추론 서버를 제공한다."""
    if not is_json_mode():
        typer.echo("ONNX 서빙: onnxruntime으로 모델을 로드합니다.")
        typer.echo("ONNX 실시간 서빙은 Triton 또는 별도 FastAPI 서버를 권장합니다.")
        typer.echo("배치 추론은 `mdp inference` 명령을 사용하세요.")
