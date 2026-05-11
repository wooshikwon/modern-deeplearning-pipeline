"""CLI 출력 유틸리티 — --format json/text 분기 처리.

글로벌 출력 포맷 상태를 관리하고, 서브커맨드 결과를
JSON 또는 Rich 테이블로 출력하는 공통 함수를 제공한다.
"""

from __future__ import annotations

import datetime
import json as json_module
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class OutputFormat(str, Enum):
    """CLI 출력 형식."""

    text = "text"
    json = "json"


OUTPUT_FORMAT_ENV = "MDP_OUTPUT_FORMAT"

# 글로벌 출력 포맷 상태. app.callback()에서 설정된다.
_output_format: OutputFormat = OutputFormat.text


def get_output_format() -> OutputFormat:
    """현재 설정된 출력 포맷을 반환한다."""
    return _output_format


def set_output_format(fmt: OutputFormat) -> None:
    """출력 포맷을 설정한다."""
    global _output_format
    _output_format = fmt
    os.environ[OUTPUT_FORMAT_ENV] = fmt.value


def is_json_mode() -> bool:
    """현재 JSON 모드인지 여부를 반환한다."""
    return _output_format == OutputFormat.json


def apply_format_override(fmt: OutputFormat | None) -> None:
    """Subcommand 레벨에서 --format이 명시되면 글로벌 포맷을 override한다.

    `mdp <cmd> --format json` 패턴을 지원한다. 명시되지 않으면(None)
    `mdp --format json <cmd>` 최상위 옵션이 그대로 유지된다.
    """
    if fmt is not None:
        set_output_format(fmt)


def apply_format_env_override() -> None:
    """Apply parent CLI output format inside subprocess entrypoints."""
    raw = os.environ.get(OUTPUT_FORMAT_ENV)
    if raw is None:
        return
    try:
        set_output_format(OutputFormat(raw))
    except ValueError:
        return


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


def schema_error_details_from_message(message: str) -> dict[str, Any]:
    """Extract schema YAML paths from settings loading validation messages."""
    errors: list[dict[str, str]] = []
    marker = "YAML path "
    for line in message.splitlines():
        stripped = line.strip()
        if marker not in stripped:
            continue
        path_and_message = stripped.split(marker, 1)[1]
        if ": " not in path_and_message:
            continue
        yaml_path, detail = path_and_message.split(": ", 1)
        errors.append({"path": yaml_path, "message": detail})
    return {"schema_errors": errors} if errors else {}


@dataclass(frozen=True)
class ModelSourcePlan:
    """CLI 모델 입력 source를 한 번에 판정한 plan."""

    kind: Literal["artifact", "pretrained"]
    command: Literal["inference", "generate", "serve", "export"]
    path: Path | None = None
    uri: str | None = None
    supports_pretrained: bool = False

    @property
    def is_pretrained(self) -> bool:
        return self.kind == "pretrained"

    @property
    def is_artifact(self) -> bool:
        return self.kind == "artifact"


_PRETRAINED_COMMANDS = ("inference", "generate")


def resolve_model_source_plan(
    run_id: str | None,
    model_dir: str | None,
    command: Literal["inference", "generate", "serve", "export"],
    pretrained: str | None = None,
) -> ModelSourcePlan:
    """run_id, model_dir, pretrained URI를 상호 배타적으로 판정한다."""
    import typer

    supports_pretrained = command in _PRETRAINED_COMMANDS
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
        if not supports_pretrained:
            raise typer.BadParameter(
                f"--pretrained는 {', '.join(_PRETRAINED_COMMANDS)} 커맨드에서만 사용할 수 있습니다."
            )
        return ModelSourcePlan(
            kind="pretrained",
            command=command,
            uri=pretrained,
            supports_pretrained=supports_pretrained,
        )

    if run_id:
        import mlflow

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
        )
        return ModelSourcePlan(
            kind="artifact",
            command=command,
            path=Path(artifact_path),
            supports_pretrained=supports_pretrained,
        )

    return ModelSourcePlan(
        kind="artifact",
        command=command,
        path=Path(model_dir),
        supports_pretrained=supports_pretrained,
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
    plan = resolve_model_source_plan(run_id, model_dir, command, pretrained=pretrained)
    return None if plan.is_pretrained else plan.path


def require_artifact_source(plan: ModelSourcePlan) -> Path:
    """artifact source가 필요한 command에서 Path를 꺼낸다."""
    import typer

    if not plan.is_artifact or plan.path is None:
        raise typer.BadParameter("artifact 모델 경로가 필요합니다.")
    return plan.path


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
