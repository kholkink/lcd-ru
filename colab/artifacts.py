"""Persistent artifacts of Colab sessions as private Kaggle datasets (no Google Drive).

name           local dir (Colab)        dataset
data           $LCDRU_TEXTS             <user>/lcd-ru-data        (split texts, data/texts)
runs/<run>     $LCDRU_RUNS/<run>        <user>/lcd-ru-runs-<run>  (quantized bases, corrections, results)

python colab/artifacts.py restore data        # download what exists (session setup)
python colab/artifacts.py save runs/p0_smoke  # publish a new version
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

USER = os.environ.get("KAGGLE_USERNAME", "kirillkholkin")
ROOTS = {"data": "LCDRU_TEXTS"}
SKIP = {"dataset-metadata.json"}


def kaggle_bin() -> str:
    return shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")


def run(cmd: list[str], timeout: int = 7200) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, e.stdout or "", f"timeout after {timeout}s")


def resolve(name: str, env: dict | None = None) -> tuple[str, str]:
    """artifact name -> (local dir, dataset id)."""
    env = env or os.environ
    if name in ROOTS:
        return env[ROOTS[name]], f"{USER}/lcd-ru-{name}"
    if name.startswith("runs/"):
        run_name = name.split("/", 1)[1]
        slug = re.sub(r"[^a-z0-9-]+", "-", f"lcd-ru-runs-{run_name}".lower()).strip("-")[:50]
        return os.path.join(env["LCDRU_RUNS"], run_name), f"{USER}/{slug}"
    raise ValueError(f"unknown artifact {name!r}: use data or runs/<name>")


def exists(dataset: str) -> bool:
    r = run([kaggle_bin(), "datasets", "status", dataset], timeout=120)
    return r.returncode == 0 and "not found" not in (r.stdout + r.stderr).lower()


def restore(name: str, env: dict | None = None, force: bool = False, tries: int = 5, wait_s: int = 30) -> str:
    """Download the artifact into its local dir unless the dir already has files. Returns a one-line report.

    A dataset that exists can still answer 404 for a few minutes after an upload (Kaggle is processing it), so an
    existing dataset is retried; only a dataset that `status` does not know is reported as absent right away."""
    local, ds = resolve(name, env)
    if not force and os.path.isdir(local) and [f for f in os.listdir(local) if f not in SKIP]:
        return f"{name}: local copy present ({local})"
    if not exists(ds):
        return f"{name}: no dataset {ds} yet"
    os.makedirs(local, exist_ok=True)
    err = ""
    for attempt in range(tries):
        r = run([kaggle_bin(), "datasets", "download", ds, "-p", local, "--unzip", "-o", "-q"])
        if r.returncode == 0 and [f for f in os.listdir(local) if f not in SKIP]:
            return f"{name}: restored from {ds} -> {local}"
        # the whole-dataset archive can stay unavailable (404) long after upload; single files download fine
        n, err = download_files(ds, local)
        if n:
            return f"{name}: restored {n} files from {ds} -> {local}"
        err = err or (r.stderr or r.stdout).strip()[-160:]
        if attempt + 1 < tries:
            time.sleep(wait_s)
    return f"{name}: RESTORE FAILED from {ds} after {tries} tries ({err})"


def list_files(ds: str) -> list[str]:
    names, token = [], None
    while True:
        cmd = [kaggle_bin(), "datasets", "files", ds, "--csv", "--page-size", "200"]
        r = run(cmd + (["--page-token", token] if token else []), timeout=300)
        if r.returncode != 0:
            return names
        token = None
        for line in r.stdout.splitlines():
            if line.startswith("Next Page Token"):
                token = line.split("=", 1)[1].strip()
            elif line and not line.startswith("name,") and "," in line:
                names.append(line.split(",")[0])
        if not token:
            return names


def download_files(ds: str, local: str) -> tuple[int, str]:
    """File-by-file download keeping the dataset's sub-directories. Returns (files downloaded, last error)."""
    n, err = 0, ""
    for f in list_files(ds):
        dest = os.path.join(local, os.path.dirname(f))
        os.makedirs(dest, exist_ok=True)
        r = run([kaggle_bin(), "datasets", "download", ds, "-f", f, "-p", dest, "-q"])
        z = os.path.join(dest, os.path.basename(f) + ".zip")
        if os.path.exists(z):  # large single files come zipped
            run(["unzip", "-o", "-q", z, "-d", dest])
            os.remove(z)
        if r.returncode == 0 and os.path.exists(os.path.join(dest, os.path.basename(f))):
            n += 1
        else:
            err = (r.stderr or r.stdout).strip()[-160:]
    return n, err


def save(name: str, env: dict | None = None, message: str = "update", exclude: tuple[str, ...] = ()) -> str:
    """Publish the local dir as a new version of its private dataset (created on first save)."""
    local, ds = resolve(name, env)
    if not os.path.isdir(local) or not [f for f in os.listdir(local) if f not in SKIP]:
        return f"{name}: nothing to save in {local}"
    stage = tempfile.mkdtemp(prefix="artifact_")
    try:
        for f in os.listdir(local):  # snapshot: the job may still be writing (periodic checkpoint saves)
            src = os.path.join(local, f)
            if f in SKIP or f in exclude or f.endswith(".tmp"):
                continue
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(stage, f))
        with open(os.path.join(stage, "dataset-metadata.json"), "w") as fh:
            json.dump({"title": ds.split("/")[1], "id": ds, "licenses": [{"name": "other"}]}, fh)  # mC4/Wikipedia texts, Qwen/Llama derivatives
        k = kaggle_bin()
        if exists(ds):
            r = run([k, "datasets", "version", "-p", stage, "-m", message, "--dir-mode", "zip", "-q"])
        else:
            r = run([k, "datasets", "create", "-p", stage, "--dir-mode", "zip", "-q"])
        if r.returncode != 0:
            return f"{name}: SAVE FAILED to {ds}: {(r.stderr or r.stdout).strip()[-300:]}"
        return f"{name}: saved to {ds} (private)"
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("restore", "save"):
        raise SystemExit(__doc__)
    fails = 0
    for name in sys.argv[2:]:
        msg = restore(name) if sys.argv[1] == "restore" else save(name)
        print(msg, flush=True)
        fails += "FAILED" in msg
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
