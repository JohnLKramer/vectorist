# Repo-sync CLI

`vectorist-cli sync` sends the Markdown files of a Git repository to a running
Vectorist server. Run it again after files change; the server skips documents
whose content has not changed.

## Install

```bash
uv sync --extra cli      # adds httpx; a server-only install does not need it
uv run vectorist-cli sync [PATH]
```

`PATH` defaults to the current directory and must be inside a Git work tree.

## What is synced

Files listed by `git ls-files` that match an include glob and no exclude glob.
Untracked and ignored files are not sent. The default include is `*.md`. A glob
matches the path relative to the repository root, and `*` also matches `/`.

A file that is not valid UTF-8 is reported as failed; the other files still sync.

## Images

When the server has image enrichment on (see [enrichment.md](enrichment.md)), each ingest result lists the images the server wants. `sync` reads each one and uploads it to the server. It does this for new and for unchanged documents, so a second run sends images the first run missed.

How a reference in a Markdown file is handled:

- A relative path is read from the file's own directory. A path that starts with `/` is read from the repository root.
- A `data:` URI with base64 data is decoded and uploaded.
- An `http` or `https` reference is not downloaded. It is counted as skipped.
- A path that leaves the repository root, an unreadable file, a malformed `data:` URI, or any other scheme is a failure for that image.
- A file must be a PNG, JPEG, GIF, or WebP image of at most 5,000,000 bytes. The CLI checks the file's first bytes, not its name.

An image that fails is reported with its document and reference, and the run exits 1 like any other failure. The documents still sync.

The run prints a second line when it handled any images:

```text
assets uploaded 3, skipped 1
```

## Flags

| Flag | Meaning |
| --- | --- |
| `--server URL` | Server base URL. Default `http://localhost:8080`. |
| `--include GLOB` | Repeatable. Replaces the configured include list. |
| `--exclude GLOB` | Repeatable. Replaces the configured exclude list. |
| `--repo-id ID` | `<host>/<owner>/<repo>` to use instead of the `origin` remote. |
| `--prune` | Delete server documents whose file is no longer in the repo. |
| `--dry-run` | Show what would be sent and pruned; change nothing. |
| `--batch-size N` | Files per request to `/v1/ingest/stream`. Default 50. |

## Configuration

Highest precedence first: flags, the `VECTORIST_SERVER_URL` environment
variable (server only), `.vectorist.toml` at the repo root, built-in defaults.

```toml
server = "http://localhost:8080"
include = ["*.md"]
exclude = ["vendor/*"]
```

An unknown key or an unparsable file stops the run with an error.

## source_uri

Each document is stored as `git:<host>/<owner>/<repo>/<relative-path>`, built
from the `origin` remote. SSH and HTTPS remotes give the same value, so a
document is the same across machines and clones. A repository with no `origin`
needs `--repo-id`.

## Pruning

With `--prune` the CLI lists the server documents under
`git:<host>/<owner>/<repo>/` and deletes those with no matching file. If no
file matches the include and exclude globs, `--prune` refuses to run, so a wrong
path or empty checkout cannot delete a repository's documents. Use `--dry-run`
to see the list first.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Everything synced. |
| 1 | Some files failed (undecodable, or the server reported `failed`). |
| 2 | Fatal: bad config, not a Git repo, no remote, server unreachable or returned an unexpected response. |

There are no retries. A fatal error stops the run.
