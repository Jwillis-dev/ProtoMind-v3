# protomind/core/versioning.py
from dataclasses import dataclass, field
from typing import Optional, Dict
import time
import uuid

@dataclass(frozen=True)
class BrainVersion:
    id: str
    parent_id: Optional[str]
    created_at: float
    notes: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def create(parent_id: Optional[str] = None, notes: str = "", tags: Dict[str, str] = None):
        return BrainVersion(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            created_at=time.time(),
            notes=notes,
            tags=tags or {}
        )
