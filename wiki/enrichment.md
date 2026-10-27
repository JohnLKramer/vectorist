# Image enrichment

An image in a document has no searchable text of its own. Image enrichment turns each image into a text description and indexes it with the chunk that holds the image, so a search for what the image shows can find that chunk.

Enrichment is off by default. Set `VECTORIST_ENRICHER` to turn it on. See [server.md](server.md) for every setting.

Enrichment covers images only. Diagram source in fenced code, such as Mermaid or PlantUML, is indexed as code, the same as before.

## How it works

1. A client ingests a document. The Markdown and HTML adapters add a media node for each `img`. The ingest response lists `pending_assets`: the `src` of each image that still needs a description.
2. The client uploads each image to `PUT /v1/documents/{document_id}/assets?ref=<src>`. The `sync` and `crawl` commands do this for you.
3. The server stores the image under its SHA-256 hash and queues one job per distinct image. The upload returns at once with `status` `queued`.
4. A worker inside the server process takes jobs one at a time and asks the provider for a description. It stores the description, then re-indexes every document that links to the image.
5. After the re-index the chunk holds the alt text and the description, and the job is `DONE`.

The same image bytes are described once, however many documents use them. An image that already has a description from the current provider is not queued again. The upload then re-indexes the document inline and returns `status` `done`.

The text stays with the image when a document is edited. If the edited document still uses the image, the next ingest keeps the stored description and asks for no new upload.

The worker marks a job `DONE` only after the linked documents are re-indexed. If the server stops partway, it returns jobs that were `RUNNING` to `PENDING` at the next start and finishes them.

## What gets enriched

The HTML adapter checks each `img` against the rules below and marks a match `SKIPPED`. A skipped image is not listed in `pending_assets`, and the server rejects an upload for it.

| Rule | Skipped when |
| :--- | :--- |
| `aria-hidden` | `aria-hidden="true"` |
| `presentation` | `role="presentation"` |
| `empty-alt` | `alt=""` |
| `too-small` | `width` or `height` is below 48 |
| `svg-data-uri` | `src` starts with `data:image/svg+xml` |
| `noise-name` | `src`, `id`, or a class contains `icon`, `logo`, `avatar`, `badge`, `sprite`, `pixel`, `tracking`, `emoji`, `banner`, or `advert`, or `src` contains `/ad/` or `/ads/` (case-insensitive) |

The Markdown adapter marks an image `SKIPPED` only when its `src` is empty. Every other Markdown image is enriched.

An image the provider describes shows up in the chunk as its alt text followed by the description. Until the description exists, the chunk holds the alt text, or the `src` when there is no alt text.

## Providers

`VECTORIST_ENRICHER` picks the provider.

| Value | Install | What it does | Needs |
| :--- | :--- | :--- | :--- |
| `none` | | Enrichment is off. The upload route returns 409 | |
| `fake` | | Returns a fixed sentence per image. For tests | |
| `ocr` | `uv sync --extra ocr` | Reads the text in the image with RapidOCR on your machine | |
| `claude` | `uv sync --extra claude` | Asks a Claude model to describe the image, its text, and for a diagram its parts and connections | `ANTHROPIC_API_KEY` |

`ocr` runs locally, so no image leaves your machine. `claude` sends each image to the Anthropic API. `VECTORIST_CLAUDE_MODEL` picks the model.

Startup fails when the provider needs something that is missing: an unknown value, the `claude` provider without `ANTHROPIC_API_KEY` or the `anthropic` package, or the `ocr` provider without the `rapidocr-onnxruntime` package. Each description is stored with the provider that wrote it. Switching providers does not describe again the images that already have a description. The current provider describes an image when a new document uses it or when it is uploaded again.

## Upload route

```bash
curl -s -X PUT 'http://127.0.0.1:8080/v1/documents/<document_id>/assets?ref=images/pool.png' \
  -H 'Content-Type: image/png' \
  --data-binary @images/pool.png
```

`ref` is the image `src` exactly as it appears in the document. The body is the raw image. The `Content-Type` must be `image/png`, `image/jpeg`, `image/gif`, or `image/webp`. A response looks like:

```json
{"asset_hash": "…", "status": "queued"}
```

`status` is `queued` when a job was created and `done` when the image already had a description and the document was re-indexed in this request. `status` is also `queued` when the image already had a description but the last ingest of the document failed. The server then does not re-index the document, and the next successful ingest of the document adds the description.

| Code | Meaning |
| :--- | :--- |
| 200 | Stored. `status` is `queued` or `done` |
| 404 | The document does not exist, it has no enrichable image with that `ref`, or it was deleted during the inline re-index |
| 409 | Enrichment is off (`VECTORIST_ENRICHER=none`) |
| 413 | The body is larger than `VECTORIST_MAX_ASSET_BYTES` |
| 415 | The `Content-Type` is not one of the four image types |
| 500 | The inline re-index failed. The image is stored and linked, and the detail has the re-index error |

## Job counts

`GET /v1/enrichment` returns the number of jobs in each state:

```json
{"pending": 0, "running": 0, "done": 3, "failed": 0}
```

Poll it to see when a batch of uploads has been described.

## Retries and failures

A provider error, or a description that is empty after trimming, counts as one failed attempt. The job returns to `PENDING`, and the worker waits `VECTORIST_ENRICH_POLL_SECONDS` before it tries the job again. It keeps trying until the job has failed `VECTORIST_ENRICH_MAX_ATTEMPTS` times. Then the job becomes `FAILED`, and the server logs the error. The worker re-indexes the linked documents, and their image nodes become `FAILED`. A failed image has no description in the chunk, so only its alt text is searchable.

The worker waits `VECTORIST_ENRICH_POLL_SECONDS` before checking again when the queue is empty. If an error other than a provider error ends the worker, the server logs it and stops processing jobs until it restarts.

To describe a `FAILED` image again, fix the cause, then run `sync` or `crawl` again. The ingest response lists a `FAILED` image in `pending_assets`, so the run uploads it again. Uploading resets a `FAILED` job to `PENDING` with no failed attempts. You can also upload the image yourself with the upload route. A re-run also sends an image that was never uploaded, such as one added to the document by an edit.

## Documents saved before enrichment

When enrichment is on, a document saved before this feature existed has image nodes with no recorded `src`. The next ingest of that document parses it again instead of skipping it, so its images gain a `src` and appear in `pending_assets`. Run `sync` or `crawl` once after you turn enrichment on.

Enrichment settings and the CLI behavior are in [server.md](server.md), [cli.md](cli.md), and [crawler.md](crawler.md).
