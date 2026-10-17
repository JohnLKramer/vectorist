from __future__ import annotations

import json
from typing import Any

import httpx

from vectorist.cli.errors import ServerError


class ServerClient:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise ServerError(f"{method} {url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise ServerError(
                f"{method} {url} returned {response.status_code}: {response.text[:200]}"
            )
        return response

    def _json(self, response: httpx.Response, url: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ServerError(f"{url} returned a non-JSON body") from exc

    def post_stream(self, documents: list[dict[str, str]]) -> list[dict[str, Any]]:
        body = "".join(json.dumps(doc) + "\n" for doc in documents)
        response = self._request(
            "POST",
            "/v1/ingest/stream",
            content=body.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        results = self._json(response, "/v1/ingest/stream")
        if not isinstance(results, list) or len(results) != len(documents):
            raise ServerError(
                f"/v1/ingest/stream returned {len(results) if isinstance(results, list) else 'a non-list'} "
                f"results for {len(documents)} documents"
            )
        return results

    def list_documents(self, prefix: str) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"source_prefix": prefix, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            response = self._request("GET", "/v1/documents", params=params)
            body = self._json(response, "/v1/documents")
            try:
                documents.extend(body["documents"])
                cursor = body["next_cursor"]
            except (KeyError, TypeError) as exc:
                raise ServerError("/v1/documents returned an unexpected body") from exc
            if cursor is None:
                return documents

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/v1/documents/{document_id}")
