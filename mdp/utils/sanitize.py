import re

_SENSITIVE_PATTERNS = re.compile(r"(token|secret|password|key|credential)", re.IGNORECASE)


def sanitize_config(config: dict) -> dict:
    """설정 딕셔너리에서 민감 정보를 마스킹한다."""
    sanitized = {}
    for k, v in config.items():
        if isinstance(v, dict):
            sanitized[k] = sanitize_config(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_config(item) if isinstance(item, dict) else item
                for item in v
            ]
        elif _SENSITIVE_PATTERNS.search(k):
            sanitized[k] = "***REDACTED***"
        else:
            sanitized[k] = v
    return sanitized
