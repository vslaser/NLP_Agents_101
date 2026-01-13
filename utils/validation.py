from __future__ import annotations
import json
from utils.exceptions import ValidationError

def parse_json_strict(text: str) -> dict:
    # First try a strict parse (ideal when the model follows instructions)
    try:
        obj = json.loads(text)
    except Exception:
        # If strict parse fails, attempt to extract the first JSON object
        # from surrounding text (handles code fences or extra commentary).
        start = text.find("{")
        if start == -1:
            raise ValidationError("Invalid JSON: no object found")

        # Find the matching closing brace using a simple stack counter.
        depth = 0
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            raise ValidationError("Invalid JSON: unbalanced braces")

        snippet = text[start:end]
        try:
            obj = json.loads(snippet)
        except Exception as e:
            raise ValidationError(f"Invalid JSON after extraction: {e}")

    if not isinstance(obj, dict):
        raise ValidationError("JSON must be an object.")
    return obj
