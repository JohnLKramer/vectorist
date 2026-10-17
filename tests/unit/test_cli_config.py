from argparse import Namespace

import pytest

from vectorist.cli.config import load_config
from vectorist.cli.errors import ConfigError
from vectorist.cli.__main__ import build_parser


def _args(**overrides):
    base = dict(
        server=None, include=None, exclude=None, repo_id=None,
        prune=False, dry_run=False, batch_size=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_parser_defaults():
    args = build_parser().parse_args(["sync"])
    assert str(args.path) == "."
    assert args.prune is False and args.dry_run is False and args.batch_size is None


def test_defaults_without_file_or_env(tmp_path):
    config = load_config(_args(), {}, tmp_path)
    assert config.server == "http://localhost:8080"
    assert config.include == ["*.md"] and config.exclude == []
    assert config.batch_size == 50


def test_precedence_flag_over_env_over_file(tmp_path):
    (tmp_path / ".vectorist.toml").write_text('server = "http://file"\ninclude = ["docs/*.md"]\n')
    assert load_config(_args(), {}, tmp_path).server == "http://file"
    env = {"VECTORIST_SERVER_URL": "http://env"}
    assert load_config(_args(), env, tmp_path).server == "http://env"
    assert load_config(_args(server="http://flag/"), env, tmp_path).server == "http://flag"
    assert load_config(_args(), env, tmp_path).include == ["docs/*.md"]
    assert load_config(_args(include=["x/*.md"]), env, tmp_path).include == ["x/*.md"]


def test_unknown_key_is_fatal(tmp_path):
    (tmp_path / ".vectorist.toml").write_text('sever = "x"\n')
    with pytest.raises(ConfigError, match="sever"):
        load_config(_args(), {}, tmp_path)


def test_malformed_toml_is_fatal(tmp_path):
    (tmp_path / ".vectorist.toml").write_text("server = = x\n")
    with pytest.raises(ConfigError):
        load_config(_args(), {}, tmp_path)


def test_bad_batch_size_is_fatal(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_args(batch_size=0), {}, tmp_path)
