import json
from typing import Any
from pathlib import Path

def load_json(file: str):
    return json.loads(Path(file).read_bytes())

def save_json(obj: Any, f: str, indent: int = 4):
    return Path(f).write_text(json.dumps(obj, indent=indent))