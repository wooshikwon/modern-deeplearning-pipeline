"""CLI 출력 유틸리티 — --format json/text 분기 처리.

글로벌 출력 포맷 상태를 관리하고, 서브커맨드 결과를
JSON 또는 Rich 테이블로 출력하는 공통 함수를 제공한다.
"""

from __future__ import annotations

import datetime
import json as json_module
import sys
from enum import Enum
from typing import Any


class OutputFormat(str, Enum):
    """CLI 출력 형식."""

    text = "text"
    json = "json"


# 글로벌 출력 포맷 상태. app.callback()에서 설정된다.
_output_format: OutputFormat = OutputFormat.text


def get_output_format() -> OutputFormat:
    """현재 설정된 출력 포맷을 반환한다."""
    return _output_format


def set_output_format(fmt: OutputFormat) -> None:
    """출력 포맷을 설정한다."""
    global _output_format
    _output_format = fmt


def is_json_mode() -> bool:
    """현재 JSON 모드인지 여부를 반환한다."""
    return _output_format == OutputFormat.json


def build_result(
    *,
    command: str,
    status: str = "success",
    error: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """공통 래퍼 구조를 포함한 결과 딕셔너리를 생성한다.

    Parameters
    ----------
    command:
        실행한 CLI 명령 이름 (train, inference, estimate, list 등).
    status:
        ``"success"`` 또는 ``"error"``.
    error:
        에러 정보. ``{"type": ..., "message": ..., "details": ...}`` 구조.
    **kwargs:
        명령별 추가 필드.
    """
    result: dict[str, Any] = {
        "status": status,
        "command": command,
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }
    if error is not None:
        result["error"] = error
    result.update(kwargs)
    return result


def build_error(
    *,
    command: str,
    error_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """에러 결과 딕셔너리를 생성한다."""
    return build_result(
        command=command,
        status="error",
        error={
            "type": error_type,
            "message": message,
            "details": details or {},
        },
    )


def resolve_model_source(
    run_id: str | None,
    model_dir: str | None,
    command: str,
    pretrained: str | None = None,
) -> Path | None:
    """run_id, model_dir, 또는 pretrained URI에서 모델 소스를 해석한다.

    세 옵션은 상호 배타적이다.  pretrained가 지정되면 None을 반환하고,
    호출부가 PretrainedResolver로 직접 로드한다.
    """
    from pathlib import Path

    import typer

    sources = [s for s in (run_id, model_dir, pretrained) if s]
    if len(sources) > 1:
        raise typer.BadParameter(
            "--run-id, --model-dir, --pretrained 중 하나만 사용할 수 있습니다."
        )
    if len(sources) == 0:
        raise typer.BadParameter(
            "--run-id, --model-dir, --pretrained 중 하나가 필요합니다."
        )

    if pretrained:
        _PRETRAINED_COMMANDS = ("inference", "generate")
        if command not in _PRETRAINED_COMMANDS:
            raise typer.BadParameter(
                f"--pretrained는 {', '.join(_PRETRAINED_COMMANDS)} 커맨드에서만 사용할 수 있습니다."
            )
        return None

    if run_id:
        import mlflow

        return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model"))
    return Path(model_dir)


def emit_result(result: dict[str, Any]) -> None:
    """_output_format에 따라 결과를 출력한다.

    - JSON 모드: stdout에 구조화된 JSON을 출력한다.
      에이전트가 stderr의 로그와 구분할 수 있도록 stdout만 사용한다.
    - Text 모드: 간소한 텍스트로 출력한다.
      (Rich 테이블은 각 서브커맨드에서 직접 처리한다.)
    """
    if _output_format == OutputFormat.json:
        print(
            json_module.dumps(result, indent=2, ensure_ascii=False, default=str),
            file=sys.stdout,
        )
    else:
        # Text 모드에서는 각 서브커맨드가 Rich 출력을 직접 처리한다.
        # 이 함수는 JSON 모드 전용으로, text 모드에서 호출되면 무시한다.
        pass
