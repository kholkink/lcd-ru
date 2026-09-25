#!/usr/bin/env python
"""Kaggle CLI helpers shared by the push/upload scripts.   python kaggle/kutil.py whoami"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def kaggle_bin() -> str:
    local = Path(sys.executable).parent / "kaggle"
    return shutil.which("kaggle") or str(local)


def username() -> str:
    """KAGGLE_USERNAME, then ~/.kaggle/kaggle.json, then `kaggle config view` (works with the access_token file)."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.exists():
        return json.loads(legacy.read_text())["username"]
    out = subprocess.run([kaggle_bin(), "config", "view"], text=True, capture_output=True)
    m = re.search(r"username:\s*(\S+)", out.stdout + out.stderr)
    if not m or m.group(1) in ("None", ""):
        raise SystemExit("Kaggle auth missing: set KAGGLE_API_TOKEN or create ~/.kaggle/access_token (chmod 600)")
    return m.group(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["whoami"]:
        print(username())
    else:
        raise SystemExit(__doc__)
