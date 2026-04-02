"""MDP 서빙 레이어 공개 API."""


def __getattr__(name: str):
    _imports = {
        "run_batch_inference": "mdp.serving.inference",
        "create_app": "mdp.serving.server",
        "create_handler": "mdp.serving.server",
    }
    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run_batch_inference",
    "create_app",
    "create_handler",
]
