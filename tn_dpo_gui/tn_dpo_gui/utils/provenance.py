from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _find_git_root(start: Path) -> Path | None:
    cursor = start.resolve()
    for parent in [cursor, *cursor.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _git_output(args: list[str], cwd: Path | None) -> str | None:
    if cwd is None:
        return None
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def runtime_provenance(start: str | Path | None = None) -> dict[str, Any]:
    anchor = Path(start).resolve() if start else Path(__file__).resolve()
    git_root = _find_git_root(anchor)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git_root": git_root.as_posix() if git_root else None,
        "git_commit": _git_output(["rev-parse", "HEAD"], git_root),
        "git_branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"], git_root),
    }
