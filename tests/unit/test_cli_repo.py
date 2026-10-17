import subprocess

import pytest

from vectorist.cli.errors import RepoError
from vectorist.cli.repo import (
    discover_files, find_repo_root, normalize_remote, repo_identity, source_uri,
)


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "docs").mkdir()
    (tmp_path / "a.md").write_text("# a")
    (tmp_path / "docs" / "b.md").write_text("# b")
    (tmp_path / "notes.txt").write_text("n")
    (tmp_path / "ignored.md").write_text("# i")
    (tmp_path / "untracked.md").write_text("# u")
    (tmp_path / ".gitignore").write_text("ignored.md\n")
    _git(tmp_path, "add", "a.md", "docs/b.md", "notes.txt", ".gitignore")
    return tmp_path


def test_discover_only_tracked_matching_files(repo):
    assert discover_files(repo, ["*.md"], []) == ["a.md", "docs/b.md"]


def test_discover_exclude_glob(repo):
    assert discover_files(repo, ["*.md"], ["docs/*"]) == ["a.md"]


def test_find_repo_root_rejects_non_repo(tmp_path):
    with pytest.raises(RepoError):
        find_repo_root(tmp_path)


def test_ssh_and_https_remotes_normalise_the_same():
    expected = "github.com/acme/widgets"
    assert normalize_remote("git@github.com:acme/widgets.git") == expected
    assert normalize_remote("https://github.com/acme/widgets.git") == expected
    assert normalize_remote("ssh://git@github.com:22/acme/widgets") == expected


def test_identity_from_origin_and_override(repo):
    _git(repo, "remote", "add", "origin", "git@github.com:acme/widgets.git")
    assert repo_identity(repo, None) == "github.com/acme/widgets"
    assert repo_identity(repo, "x/y/z") == "x/y/z"
    assert source_uri("x/y/z", "docs/b.md") == "git:x/y/z/docs/b.md"


def test_identity_without_origin_fails(repo):
    with pytest.raises(RepoError, match="--repo-id"):
        repo_identity(repo, None)
