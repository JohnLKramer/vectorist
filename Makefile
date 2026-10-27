PG_URL := postgresql://vectorist:vectorist@localhost:55432/vectorist

.PHONY: test test-pg test-live test-embedder test-crawl test-enricher

test:
	uv run pytest

test-pg:
	trap 'docker compose -f docker-compose.test.yml down -v' EXIT; \
	docker compose -f docker-compose.test.yml up -d --wait && \
	VECTORIST_PG_URL=$(PG_URL) uv run pytest -m backends

test-live:
	uv run pytest -m live

test-embedder:
	uv run pytest -m embedder -o addopts=""

test-crawl:
	uv run playwright install chromium
	uv run pytest -m crawl

test-enricher:
	uv run pytest -m enricher -o addopts=""
