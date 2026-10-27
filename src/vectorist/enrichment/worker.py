from __future__ import annotations

import asyncio
import logging

from vectorist.cache.protocol import CacheProtocol
from vectorist.enrichment.protocol import ImageEnricherProtocol
from vectorist.ir.models import EnrichmentStatus, MediaNode
from vectorist.pipeline.ingest import IngestService

logger = logging.getLogger(__name__)


class EnrichmentWorker:
    """Polls the enrichment job queue, describes images, and re-indexes documents.

    Provider errors and empty descriptions fail the job through ``fail_job``.
    After a failure that will be retried, the worker waits
    ``retry_delay_seconds`` (default: ``poll_seconds``) before the next claim.
    Any other error propagates out of ``run_once`` and ends the polling task,
    which is logged by a done callback.
    """

    def __init__(
        self,
        cache: CacheProtocol,
        ingest: IngestService,
        enricher: ImageEnricherProtocol,
        poll_seconds: float,
        max_attempts: int,
        retry_delay_seconds: float | None = None,
    ) -> None:
        self._cache = cache
        self._ingest = ingest
        self._enricher = enricher
        self._poll_seconds = poll_seconds
        self._max_attempts = max_attempts
        self._retry_delay_seconds = (
            poll_seconds if retry_delay_seconds is None else retry_delay_seconds
        )
        self._task: asyncio.Task[None] | None = None

    async def recover(self) -> None:
        """Return jobs left ``RUNNING`` by a previous process to ``PENDING``."""
        count = await self._cache.requeue_running_jobs()
        if count:
            logger.info("requeued %d running enrichment jobs", count)

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("enrichment worker already started")
        self._task = asyncio.create_task(self._poll(), name="enrichment-worker")
        self._task.add_done_callback(_log_task_end)

    async def stop(self) -> None:
        """Cancel the polling task and wait for it to end.

        An exception that already ended the task was logged by the done callback.
        """
        task = self._task
        if task is None:
            return
        self._task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Already logged by _log_task_end.
            pass

    async def _poll(self) -> None:
        while True:
            if not await self.run_once():
                await asyncio.sleep(self._poll_seconds)

    async def run_once(self) -> bool:
        """Claim and process one job. Returns ``False`` when the queue is empty."""
        asset_hash = await self._cache.claim_job()
        if asset_hash is None:
            return False

        asset = await self._cache.get_asset(asset_hash)
        if asset is None:
            logger.info(
                "enrichment asset %s is gone (document deleted); skipping job",
                asset_hash,
            )
            return True

        provider_id = self._enricher.provider_id
        if asset_hash in await self._cache.get_enrichments([asset_hash], provider_id):
            # A previous run stored the description but stopped before the
            # re-index finished; the job was requeued by recover().
            await self._reindex_linked_and_complete(asset_hash)
            return True

        alt_text = await self._alt_text(asset_hash)
        try:
            text = await asyncio.to_thread(
                self._enricher.describe, asset.data, asset.content_type, alt_text
            )
            if not text.strip():
                raise ValueError(
                    f"enricher {self._enricher.provider_id} returned no text"
                )
        except Exception as exc:
            await self._fail(asset_hash, exc)
            return True

        document_ids = await self._cache.document_ids_for_asset(asset_hash)
        if not document_ids:
            logger.info(
                "enrichment asset %s has no linked documents (deleted while "
                "describing); dropping result",
                asset_hash,
            )
            return True

        await self._cache.put_enrichment(asset_hash, provider_id, text)
        await self._reindex_linked_and_complete(asset_hash)
        return True

    async def _reindex_linked_and_complete(self, asset_hash: str) -> None:
        """Re-index every document linked to ``asset_hash``, then mark the job done.

        The links are read after the description is stored, so a document
        linked meanwhile is included. The job stays ``RUNNING`` until every
        re-index finishes, so a crash or cancellation leaves it for
        ``recover()``.
        """
        await self._reindex(await self._cache.document_ids_for_asset(asset_hash))
        await self._cache.complete_job(asset_hash)

    async def _fail(self, asset_hash: str, exc: Exception) -> None:
        status = await self._cache.fail_job(asset_hash, str(exc), self._max_attempts)
        logger.error(
            "enrichment of asset %s with provider %s failed (job now %s, "
            "max_attempts=%d): %s",
            asset_hash,
            self._enricher.provider_id,
            status,
            self._max_attempts,
            exc,
            exc_info=exc,
        )
        if status == "FAILED":
            await self._reindex(await self._cache.document_ids_for_asset(asset_hash))
            return
        # The job is back at the head of the queue; wait so the attempts are
        # spread out instead of burning in milliseconds.
        await asyncio.sleep(self._retry_delay_seconds)

    async def _reindex(self, document_ids: list[str]) -> None:
        for document_id in document_ids:
            result = await self._ingest.reindex(document_id)
            if result is not None and result.status == "skipped":
                logger.info(
                    "document %s was not re-indexed after enrichment (not "
                    "INDEXED); its next ingest applies the description",
                    document_id,
                )
            elif result is not None and result.status == "failed":
                logger.error(
                    "re-index of document %s after enrichment failed: %s",
                    document_id,
                    result.error,
                )

    async def _alt_text(self, asset_hash: str) -> str:
        """Alt text of the first image node linked to ``asset_hash``, or ``""``."""
        document_ids = await self._cache.document_ids_for_asset(asset_hash)
        if not document_ids:
            return ""
        document_id = document_ids[0]
        snapshot = await self._cache.get_ir_snapshot(document_id)
        if snapshot is None:
            return ""
        links = await self._cache.get_document_assets(document_id)
        refs = {ref for ref, linked in links.items() if linked == asset_hash}
        for node in snapshot.nodes.values():
            if not isinstance(node, MediaNode):
                continue
            if node.enrichment_status == EnrichmentStatus.SKIPPED:
                continue
            if node.metadata.get("ref") in refs:
                return node.alt_text
        return ""


def _log_task_end(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "enrichment worker stopped on an unexpected error; no further jobs "
            "will be processed",
            exc_info=exc,
        )
