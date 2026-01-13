from __future__ import annotations

def render(template: str, **kwargs) -> str:
    # Keep templates simple and explicit; avoid arbitrary eval.
    return template.format(**kwargs)
