from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    SECTION = "SECTION"
    PARAGRAPH = "PARAGRAPH"
    CODE_BLOCK = "CODE_BLOCK"
    TABLE = "TABLE"
    MEDIA = "MEDIA"
    LIST = "LIST"


class EnrichmentStatus(str, Enum):
    PENDING = "PENDING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class IRNodeBase(BaseModel):
    node_id: str
    node_type: NodeType
    parent_id: str | None
    children_ids: list[str]
    breadcrumbs: list[str]
    line_range: tuple[int, int]
    raw_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionNode(IRNodeBase):
    node_type: Literal[NodeType.SECTION] = NodeType.SECTION
    level: int
    title: str


class ParagraphNode(IRNodeBase):
    node_type: Literal[NodeType.PARAGRAPH] = NodeType.PARAGRAPH


class ListNode(IRNodeBase):
    node_type: Literal[NodeType.LIST] = NodeType.LIST
    ordered: bool = False


class CodeBlockNode(IRNodeBase):
    node_type: Literal[NodeType.CODE_BLOCK] = NodeType.CODE_BLOCK
    language: str
    detection_source: str


class TableNode(IRNodeBase):
    node_type: Literal[NodeType.TABLE] = NodeType.TABLE
    caption: str | None = None
    headers: list[str]
    column_types: list[str] = Field(default_factory=list)
    rows: list[list[str]]


class MediaNode(IRNodeBase):
    node_type: Literal[NodeType.MEDIA] = NodeType.MEDIA
    media_type: str
    alt_text: str = ""
    caption: str = ""
    asset_uri_or_hash: str = ""
    enrichment_status: EnrichmentStatus = EnrichmentStatus.PENDING


IRNode = Annotated[
    SectionNode
    | ParagraphNode
    | ListNode
    | CodeBlockNode
    | TableNode
    | MediaNode,
    Field(discriminator="node_type"),
]


class IRDocument(BaseModel):
    source_uri: str
    root_id: str
    nodes: dict[str, IRNode]
