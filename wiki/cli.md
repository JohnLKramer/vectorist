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
