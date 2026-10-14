from dataclasses import dataclass
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PARSED = "PARSED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


@dataclass
class DocumentRecord:
    id: str
    source_uri: str
    content_hash: str
    raw_content: str
    status: DocumentStatus
    error_message: str | None = None
