"""Where the data lives.  One root, no absolute paths in the code.

Resolution order (first hit wins):
  1. an explicit path (the CLI's ``--data`` flag)
  2. the ``FLAKECV_DATA`` environment variable
  3. ``data = "..."`` under ``[paths]`` in a ``flakecv.toml`` found in the
     current directory or any parent; a relative path is taken relative to
     that file, so the repo can be moved or cloned anywhere
  4. ``./data`` in the current directory

If the chosen directory does not exist, fail with a message that says which
rule chose it and how to fix it.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

CONFIG_NAME = "flakecv.toml"
ENV_VAR = "FLAKECV_DATA"


class DataRootError(FileNotFoundError):
    pass


def find_config(start: Path | None = None) -> Path | None:
    d = Path(start or Path.cwd()).resolve()
    for p in (d, *d.parents):
        f = p / CONFIG_NAME
        if f.is_file():
            return f
    return None


def data_root(explicit: str | os.PathLike | None = None, start: Path | None = None) -> Path:
    if explicit:
        p, why = Path(explicit).expanduser(), "the --data flag"
    elif os.environ.get(ENV_VAR):
        p, why = Path(os.environ[ENV_VAR]).expanduser(), f"the {ENV_VAR} environment variable"
    else:
        cfg = find_config(start)
        rel = None
        if cfg is not None:
            with open(cfg, "rb") as fh:
                rel = tomllib.load(fh).get("paths", {}).get("data")
        if rel is not None:
            p, why = (cfg.parent / rel), f"[paths] data in {cfg}"
        else:
            p, why = Path(start or Path.cwd()) / "data", "the default ./data"
    p = p.resolve()
    if not p.is_dir():
        raise DataRootError(
            f"Data directory not found: {p}\n"
            f"  (chosen by {why})\n"
            f"Fix: pass --data <folder>, set {ENV_VAR}, or put\n"
            f'  [paths]\n  data = "data"\n'
            f"in a {CONFIG_NAME} next to your data folder.")
    return p
