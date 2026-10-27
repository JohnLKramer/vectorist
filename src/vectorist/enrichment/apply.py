from __future__ import annotations

from vectorist.ir.models import EnrichmentStatus, IRDocument, MediaNode


class LegacySnapshotError(ValueError):
    """An IR snapshot has image nodes saved before image refs were recorded."""


def has_legacy_media(ir: IRDocument) -> bool:
    """True when a non-skipped image node in ``ir`` has no ``metadata["ref"]``."""
    return any(
        isinstance(node, MediaNode)
        and node.enrichment_status != EnrichmentStatus.SKIPPED
        and "ref" not in node.metadata
        for node in ir.nodes.values()
    )


def media_ref(ir: IRDocument, node: MediaNode) -> str:
    """Return the node's image ref, failing loudly on a legacy snapshot."""
    ref = node.metadata.get("ref")
    if ref is None:
        raise LegacySnapshotError(
            f"IR snapshot for {ir.source_uri} predates image refs "
            f"(node {node.node_id}); re-ingest the document"
        )
    return ref


def apply_enrichments(
    ir: IRDocument,
    links: dict[str, str],
    texts: dict[str, str],
    job_status: dict[str, str],
) -> IRDocument:
    """Return a copy of ``ir`` with stored image text applied to linked media nodes.

    ``links`` maps a node's ``metadata["ref"]`` to an asset hash, ``texts`` maps an
    asset hash to its description, and ``job_status`` maps an asset hash to its job
    status. Caption and raw content of linked nodes are rebuilt from ``alt_text``,
    the ref and the stored text, so applying the same inputs again gives the same
    document and text that is no longer stored is dropped.
    """
    result = ir.model_copy(deep=True)
    for node in result.nodes.values():
        if not isinstance(node, MediaNode):
            continue
        if node.enrichment_status == EnrichmentStatus.SKIPPED:
            continue
        ref = media_ref(result, node)
        asset_hash = links.get(ref)
        if asset_hash is None:
            continue
        node.asset_uri_or_hash = f"sha256:{asset_hash}"
        if asset_hash in texts:
            text = texts[asset_hash]
            node.caption = text
            node.raw_content = f"{node.alt_text}\n{text}" if node.alt_text else text
            node.enrichment_status = EnrichmentStatus.DONE
            continue
        node.caption = ""
        node.raw_content = node.alt_text or ref
        if job_status.get(asset_hash) == "FAILED":
            node.enrichment_status = EnrichmentStatus.FAILED
        else:
            node.enrichment_status = EnrichmentStatus.PENDING
    return result
