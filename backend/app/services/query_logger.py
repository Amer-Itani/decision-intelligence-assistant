"""JSON Lines logging for analysis requests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings


class QueryLogger:
    """Persist request/response audit events for review and debugging."""

    def __init__(self, settings: Settings) -> None:
        self._log_path = Path(settings.log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        """Append one JSON event to the request log."""

        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            **self._serialize(payload),
        }
        with self._log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, dict):
            return {key: self._serialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        return value
