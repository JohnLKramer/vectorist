from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vectorist.cli.client import ServerClient
from vectorist.cli.config import CliConfig
from vectorist.cli.errors import CliError
from vectorist.cli.repo import discover_files, repo_identity, source_uri, uri_prefix


@dataclass
class SyncSummary:
    sent: int = 0
    ingested: int = 0
    unchanged: int = 0
    failures: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    dry_run: bool = False


def run_sync(config: CliConfig, client: ServerClient, root: Path) -> SyncSummary:
    identity = repo_identity(root, config.repo_id)
    files = discover_files(root, config.include, config.exclude)
    if config.prune and not files:
        raise CliError(
            "--prune refused: no files matched, which would delete every document "
            f"under {uri_prefix(identity)}"
        )
    summary = SyncSummary(dry_run=config.dry_run)
    uris = {path: source_uri(identity, path) for path in files}

    documents: list[dict[str, str]] = []
    for path in files:
        try:
            content = (root / path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            summary.failures.append(f"{path}: not valid UTF-8 ({exc.reason})")
            continue
        documents.append({"source_uri": uris[path], "content": content})
    summary.sent = len(documents)

    if not config.dry_run:
        for start in range(0, len(documents), config.batch_size):
            batch = documents[start : start + config.batch_size]
            for result in client.post_stream(batch):
                status = result.get("status")
                if status == "failed":
                    error = result.get("error") or {}
                    summary.failures.append(
                        f"{result.get('source_uri')}: {error.get('message', 'failed')}"
                    )
                elif status == "skipped":
                    summary.unchanged += 1
                else:
                    summary.ingested += 1

    if config.prune:
        keep = set(uris.values())
        for document in client.list_documents(uri_prefix(identity)):
            if document["source_uri"] in keep:
                continue
            if not config.dry_run:
                client.delete_document(document["id"])
            summary.pruned.append(document["source_uri"])
    return summary
