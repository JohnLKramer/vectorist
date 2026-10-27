# Documentation-site crawler

`vectorist-cli crawl URL` crawls a documentation site in a headless browser and
sends each page to a running Vectorist server as HTML. JavaScript-rendered pages
work, because the page is indexed as the browser sees it after loading.

## Install

```bash
uv sync --extra crawl
uv run playwright install chromium   # one-time browser download
uv run vectorist-cli crawl https://docs.example.com/guide/
```

If the browser is missing, the command stops and tells you to run
`playwright install chromium`. It does not fall back to plain HTTP fetches.

## Scope

A page is crawled only if it has the same scheme and host as the start URL and
its path is under the start URL's path (`/guide` matches `/guide` and
`/guide/x`, not `/guide-old`). If the last segment of the start URL has a file
extension, such as `/guide/index.html`, the scope is its directory (`/guide`).
Links to files with non-HTML extensions are ignored.

Pages are found from `sitemap.xml` (or a `Sitemap:` line in `robots.txt`)
when the site has one with URLs in scope. Sitemap index files are followed. When
there is no usable sitemap, the crawler follows links found in the rendered
pages, breadth first from the start URL. `robots.txt` is honoured for the
`vectorist` user agent; disallowed pages are skipped and counted.

## Images

When the server has image enrichment on (see [enrichment.md](enrichment.md)), the crawler downloads the images the server asks for and uploads them. It downloads them with the same HTTP client it uses for `robots.txt` and sitemaps, not with the browser, so an image that needs a logged-in browser session or a script to load is not fetched.

- Images may be on any host. They are not limited to the crawl scope, and `robots.txt` is not applied to them.
- A relative reference is resolved against the page URL. A `data:` URI is decoded.
- A download under 2 KB is skipped and counted. This drops tracking pixels and small decorations that the page markup did not mark.
- A failed download, an unsupported file type, or an image over 5,000,000 bytes is reported as a failure for that page. The page itself still syncs.

The summary adds `assets uploaded N, skipped N` when any images were handled.

## Flags

| Flag | Meaning |
| --- | --- |
| `URL` | Start URL. May come from `[crawl] url` in the config file. |
| `--server URL` | Server base URL. Default `http://localhost:8080`. |
| `--max-pages N` | Stop after N page attempts. Default 500. |
| `--delay SECONDS` | Pause after each page load, per worker. Default 0.5. |
| `--concurrency N` | Pages rendered at once, each with its own browser. Default 1. |
| `--keep-query` | Keep query strings in `source_uri`. Off by default. |
| `--prune` | Delete server documents in scope that this crawl did not see. |
| `--dry-run` | Crawl and render, but send and delete nothing. |
| `--batch-size N` | Pages per request to `/v1/ingest/stream`. Default 50. |

## Configuration

Flags, then `VECTORIST_SERVER_URL` (server only), then `.vectorist.toml` in the
current directory, then defaults.

```toml
server = "http://localhost:8080"

[crawl]
url = "https://docs.example.com/guide/"
max_pages = 200
delay = 1.0
concurrency = 2
```

An unknown key, a wrong type, or an unparsable file stops the run.

## source_uri

The final URL after redirects, with the fragment removed, the trailing slash
removed, and the query removed unless `--keep-query` is set. A page that
redirects outside the scope is skipped and counted.

## Pruning

`--prune` lists the server documents under the scope and deletes those the crawl
did not see. It refuses to run, and deletes nothing, when:

- no pages were fetched,
- the crawl stopped at `--max-pages`, or
- any page failed to render.

In each case the page set is incomplete, so absence does not mean the page was
removed. Use `--dry-run` to preview the list.

## Failures and exit codes

A page that times out, returns a non-2xx status, or renders an empty body is
reported and the crawl continues. A site that needs a login will usually show
up as empty or error pages. Authentication is not supported.

| Code | Meaning |
| --- | --- |
| 0 | Everything synced. |
| 1 | Some pages failed, or the server rejected them. |
| 2 | Fatal: bad config, missing browser, unreadable `robots.txt` or sitemap, or the server is unreachable. |

There are no retries.
