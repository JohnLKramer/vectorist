from typing import Protocol

from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.ir.models import IRDocument


class CacheProtocol(Protocol):
    async def get_document_by_uri(self, source_uri: str) -> DocumentRecord | None: ...

    async def get_document(self, document_id: str) -> DocumentRecord | None: ...

    async def delete_document(self, document_id: str) -> None: ...

    async def upsert_document(self, doc: DocumentRecord) -> None: ...

    async def save_ir_snapshot(self, document_id: str, ir_tree: IRDocument) -> None: ...

    async def get_ir_snapshot(self, document_id: str) -> IRDocument | None: ...

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None: ...
