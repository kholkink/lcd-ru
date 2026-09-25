import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config  # noqa: E402


def config_from_cli(description: str) -> dict:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="a.b.c=value")
    a = ap.parse_args()
    return load_config(a.config, a.overrides)
