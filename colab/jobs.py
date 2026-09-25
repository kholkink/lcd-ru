#!/usr/bin/env python
"""Local side of the Colab worker (colab/worker.py, ported from ~/casmi26). Queue = private Kaggle dataset
<user>/lcd-ru-jobs (jobs.json); status = <user>/lcd-ru-colab-status-<letter> (status.json, logs/<id>.log,
results/<id>/); artifacts (data, runs/<name>) = private datasets, see colab/artifacts.py.

    python colab/jobs.py add --id p0-smoke --gpu --timeout-h 2 --restore runs/p0_smoke --save runs/p0_smoke \
        --save-every 20 --results '$LCDRU_RUNS/p0_smoke*/*.jsonl' \
        --cmd "python scripts/smoke.py --config configs/p0_smoke.yaml"
    python colab/jobs.py list                     # local queue
    python colab/jobs.py status [--worker A]      # what the worker reported last (state, current job, finished)
    python colab/jobs.py logs --id ctr-v1         # log tail of a job (running jobs: refreshed every ~10 min)
    python colab/jobs.py results --id ctr-v1      # download + list the job's small result files
    python colab/jobs.py pull --what runs/ctr-v1  # download an artifact dataset -> build/artifacts/runs/ctr-v1
    python colab/jobs.py wait --id ctr-v1         # block until the job finished (then prints its log tail)
    python colab/jobs.py cancel --id ctr-v1       # the worker kills it at its next check
    python colab/jobs.py rm|disable|enable --id X ; stop|resume [--worker A]
    python colab/jobs.py front --ids a b c        # move jobs to the head of the queue, in this order

Every mutating command publishes the queue (the worker sees it within its poll period plus Kaggle processing, ~1-3
min). Jobs carry the code version from the last `bash kaggle/push_code.sh`, so the worker waits for that upload.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "kaggle"))
sys.path.insert(0, os.path.join(REPO, "colab"))
import artifacts  # noqa: E402
from kutil import kaggle_bin  # noqa: E402

QDIR = os.path.join(REPO, "build", "jobs")
QFILE = os.path.join(QDIR, "jobs.json")
SDIR = os.path.join(REPO, "build", "jobs_status")
VFILE = os.path.join(REPO, "build", "code_version.txt")


def load() -> dict:
    if os.path.exists(QFILE):
        with open(QFILE) as f:
            return json.load(f)
    return {"jobs": [], "cancel": [], "stop_workers": []}


def save(q: dict) -> None:
    os.makedirs(QDIR, exist_ok=True)
    with open(QFILE, "w") as f:
        json.dump(q, f, indent=1)


def publish(q: dict, user: str) -> None:
    save(q)
    with open(os.path.join(QDIR, "dataset-metadata.json"), "w") as f:
        json.dump({"title": "lcd-ru-jobs", "id": f"{user}/lcd-ru-jobs", "licenses": [{"name": "CC0-1.0"}]}, f)
    k = kaggle_bin()
    r = subprocess.run(
        [k, "datasets", "version", "-p", QDIR, "-m", "queue update", "-q"], capture_output=True, text=True
    )
    if r.returncode != 0:
        r = subprocess.run([k, "datasets", "create", "-p", QDIR, "-q"], capture_output=True, text=True)
    print("queue published" if r.returncode == 0 else f"publish FAILED: {(r.stderr or r.stdout)[-300:]}")


def fetch_status(user: str, worker: str) -> tuple[dict | None, str]:
    d = os.path.join(SDIR, worker.lower())
    subprocess.run(["rm", "-rf", d])
    r = subprocess.run(
        [kaggle_bin(), "datasets", "download", f"{user}/lcd-ru-colab-status-{worker.lower()}", "-p", d, "--unzip", "-o", "-q"],
        capture_output=True, text=True,
    )  # fmt: skip
    for z in glob.glob(os.path.join(d, "*.zip")):  # --dir-mode zip uploads: unpack nested archives if Kaggle kept them
        subprocess.run(["unzip", "-o", "-q", z, "-d", os.path.splitext(z)[0]], capture_output=True)
    p = os.path.join(d, "status.json")
    if r.returncode != 0 or not os.path.exists(p):
        return None, d
    with open(p) as f:
        return json.load(f), d


def show_status(st: dict | None, worker: str) -> None:
    if st is None:
        print(f"worker {worker}: no status yet (worker never started, or Kaggle is still processing)")
        return
    cur = st.get("current")
    print(f"worker {st['worker']} [{st['state']}] at {st['time_utc']} UTC on {st.get('gpu') or 'CPU'}")
    if cur:
        print(f"  running {cur['id']} since {cur['start']} ({cur.get('elapsed_min', 0)} min): {cur['cmd'][:150]}")
    fin = st.get("finished", {})
    for jid, f in sorted(fin.items(), key=lambda kv: kv[1].get("end", ""))[-15:]:
        mark = "ok  " if f["rc"] == 0 else f"FAIL rc={f['rc']}"
        print(f"  {mark} {jid} ({f.get('minutes')} min, {f.get('end')}){' ' + f['note'] if f.get('note') else ''}")


def read_log(d: str, jid: str, lines: int) -> str:
    p = os.path.join(d, "logs", f"{jid}.log")
    if not os.path.exists(p):
        return f"(no log for {jid} yet)"
    with open(p) as f:
        return "".join(f.readlines()[-lines:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "cmd",
        choices=["add", "rm", "disable", "enable", "cancel", "list", "publish", "status", "logs", "results", "wait",
                 "pull", "stop", "resume", "front"],
    )  # fmt: skip
    ap.add_argument("--id")
    ap.add_argument("--ids", nargs="*", default=[], help="front: several job ids, in the order wanted")
    ap.add_argument(
        "--cmd", dest="shell", help="bash command, run in the code dir with the LCDRU_* env"
    )
    ap.add_argument("--worker", default="A")
    ap.add_argument("--after", nargs="*", default=[], help="job ids that must have succeeded first")
    ap.add_argument("--results", nargs="*", default=[], help="globs of small files to ship back (<= 25 MB each)")
    ap.add_argument("--gpu", action="store_true", help="needs a GPU runtime")
    ap.add_argument("--timeout-h", type=float, default=None)
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs for this job")
    ap.add_argument("--setup-again", action="store_true", help="rerun session_setup.sh before this job")
    ap.add_argument("--restore", nargs="*", default=[], help="artifacts to download first (runs/<name>, data)")
    ap.add_argument(
        "--save", nargs="*", default=[], help="artifacts to publish after the job (data, runs/<name>)"
    )
    ap.add_argument(
        "--save-every", type=float, default=None, help="minutes between runs/* checkpoint saves while running"
    )
    ap.add_argument("--what", help="pull: artifact name (data, runs/<name>)")
    ap.add_argument("--lines", type=int, default=60)
    ap.add_argument("--poll", type=float, default=90, help="wait: seconds between status checks")
    ap.add_argument("--user", default="kirillkholkin")
    ap.add_argument("--no-publish", action="store_true")
    a = ap.parse_args()
    q = load()
    q.setdefault("cancel", [])
    q.setdefault("stop_workers", [])

    if a.cmd == "add":
        if not a.id or not a.shell:
            raise SystemExit("add needs --id and --cmd")
        job = {"id": a.id, "worker": a.worker.upper(), "cmd": a.shell, "after": a.after, "results": a.results,
               "gpu": a.gpu, "timeout_h": a.timeout_h, "env": dict(e.split("=", 1) for e in a.env),
               "setup_again": a.setup_again, "restore": a.restore, "save": a.save, "save_every": a.save_every,
               "added": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        for name in a.restore + a.save:
            artifacts.resolve(name, {"LCDRU_TEXTS": "", "LCDRU_RUNS": ""})
        if os.path.exists(VFILE):
            job["code_version"] = open(VFILE).read().strip()
        q["jobs"] = [j for j in q["jobs"] if j["id"] != a.id] + [job]
        q["cancel"] = [c for c in q["cancel"] if c != a.id]
        print(f"queued {a.id} for worker {job['worker']} (code {job.get('code_version', '?')})")
    elif a.cmd in ("rm", "disable", "enable", "cancel"):
        hit = [j for j in q["jobs"] if j["id"] == a.id]
        if not hit:
            raise SystemExit(f"no job {a.id}")
        if a.cmd == "rm":
            q["jobs"] = [j for j in q["jobs"] if j["id"] != a.id]
        elif a.cmd == "cancel":
            q["cancel"] = sorted(set(q["cancel"]) | {a.id})
            hit[0]["disabled"] = True
        else:
            hit[0]["disabled"] = a.cmd == "disable"
        print(f"{a.cmd}: {a.id}")
    elif a.cmd == "front":  # move jobs to the head of the queue (workers take the first runnable job)
        ids = [i for i in (a.ids or [a.id]) if i]
        missing = [i for i in ids if i not in {j["id"] for j in q["jobs"]}]
        if missing:
            raise SystemExit(f"no jobs {missing}")
        first = [next(j for j in q["jobs"] if j["id"] == i) for i in ids]
        q["jobs"] = first + [j for j in q["jobs"] if j["id"] not in ids]
        print(f"front: {ids}")
    elif a.cmd in ("stop", "resume"):
        st = {w.upper() for w in q["stop_workers"]}
        q["stop_workers"] = sorted(st | {a.worker.upper()} if a.cmd == "stop" else st - {a.worker.upper()})
        print(f"stop_workers = {q['stop_workers']}")
    elif a.cmd == "list":
        for j in q["jobs"]:
            flags = [f for f, on in (("gpu", j.get("gpu")), ("disabled", j.get("disabled"))) if on]
            after = f" after {j['after']}" if j.get("after") else ""
            print(f"{j['worker']} {j['id']}{after} {flags or ''}\n    {j['cmd'][:160]}")
        return
    elif a.cmd == "status":
        show_status(fetch_status(a.user, a.worker)[0], a.worker)
        return
    elif a.cmd == "logs":
        st, d = fetch_status(a.user, a.worker)
        show_status(st, a.worker)
        print(read_log(d, a.id, a.lines))
        return
    elif a.cmd == "results":
        st, d = fetch_status(a.user, a.worker)
        rd = os.path.join(d, "results", a.id)
        files = sorted(glob.glob(os.path.join(rd, "*")))
        print(f"{rd}:" if files else f"no results for {a.id}")
        for p in files:
            print(f"  {os.path.basename(p)} ({os.path.getsize(p):,} B)")
        return
    elif a.cmd == "pull":
        env = {"LCDRU_TEXTS": os.path.join(REPO, "build", "artifacts", "texts"),
               "LCDRU_RUNS": os.path.join(REPO, "build", "artifacts", "runs")}  # fmt: skip
        print(artifacts.restore(a.what, env, force=True))
        return
    elif a.cmd == "wait":
        t0 = time.time()
        while True:
            st, d = fetch_status(a.user, a.worker)
            f = (st or {}).get("finished", {}).get(a.id)
            if f is not None:
                print(f"{a.id}: rc={f['rc']} after {f.get('minutes')} min {f.get('note') or ''}")
                print(read_log(d, a.id, a.lines))
                sys.exit(0 if f["rc"] == 0 else 1)
            cur = (st or {}).get("current") or {}
            print(f"[{(time.time() - t0) / 60:.0f} min] worker {(st or {}).get('state', '?')}, running "
                  f"{cur.get('id', '-')} ({cur.get('elapsed_min', 0)} min)", flush=True)  # fmt: skip
            time.sleep(a.poll)
    if not a.no_publish:
        publish(q, a.user)
    else:
        save(q)


if __name__ == "__main__":
    main()
