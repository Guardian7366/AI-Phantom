import json
import os
from datetime import datetime
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: str, data: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

