import json

from fastapi import APIRouter, Request
from pydantic import ValidationError

from vectorist.api.schemas import IngestRequest, IngestResponse
from vectorist.pipeline.ingest import IngestResult

router = APIRouter()


def _response(result: IngestResult) -> IngestResponse:
    return IngestResponse(
        source_uri=result.source_uri,
        document_id=result.document_id,
        status=result.status,
        error=result.error,
    )


def _failed_line(source_uri: str, message: str) -> IngestResponse:
    return IngestResponse(
        source_uri=source_uri,
        document_id=None,
        status="failed",
        error={"code": "INVALID_LINE", "message": message},
    )


@router.post("/v1/ingest", response_model=IngestResponse)
async def ingest_document(body: IngestRequest, request: Request) -> IngestResponse:
    result = await request.app.state.ingest.ingest_document(
        body.source_uri,
        body.content,
        content_type=body.content_type,
    )
    return _response(result)


@router.post("/v1/ingest/stream", response_model=list[IngestResponse])
async def ingest_stream(request: Request) -> list[IngestResponse]:
    raw = await request.body()
    text = raw.decode("utf-8")
    results: list[IngestResponse] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            body = IngestRequest.model_validate(payload)
        except json.JSONDecodeError as exc:
            results.append(_failed_line("", str(exc)))
            continue
        except ValidationError as exc:
            source_uri = ""
            if isinstance(payload, dict) and isinstance(payload.get("source_uri"), str):
                source_uri = payload["source_uri"]
            results.append(_failed_line(source_uri, str(exc)))
            continue
        result = await request.app.state.ingest.ingest_document(
            body.source_uri,
            body.content,
            content_type=body.content_type,
        )
        results.append(_response(result))
    return results
