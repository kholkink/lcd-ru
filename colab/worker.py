"""Colab worker (ported from ~/casmi26): runs shell jobs queued in the private Kaggle dataset <user>/lcd-ru-jobs;
everything persistent lives in private Kaggle datasets (no Google Drive).

One cell per Colab session (colab/worker.ipynb puts the tokens and the code in place first):

    %run /content/code/colab/worker.py --worker A

Jobs are added from the local machine with `python colab/jobs.py add ...` (publishes a new queue version). The loop:

  * restores its own history (finished jobs, log tails, results) from its status dataset
    <user>/lcd-ru-colab-status-<letter>, so a new session never reruns finished work;
  * session setup once (colab/session_setup.sh: pip, `data` artifact from Kaggle, see colab/artifacts.py);
  * polls jobs.json every --poll-minutes; picks the first enabled job of its letter that is not finished, whose
    `after` jobs succeeded and whose GPU need the runtime meets;
  * refreshes the code dataset before every job (waits for the code version the job was queued with);
  * restores the job's `restore` artifacts, runs `bash -c <cmd>` in the code dir with the LCDRU_* environment,
    saves `runs/*` artifacts every `save_every` minutes while it runs (checkpoints survive a dead runtime) and all
    `save` artifacts at the end (runs/* even on failure, the rest only on success);
  * publishes the status dataset at start, every --report-minutes while running, and at the end: status.json,
    logs/<id>.log (tail), results/<id>/ (small files listed in `results`);
  * cancels a running job whose id appears in the queue's `cancel` list, kills it after `timeout_h`;
  * releases the runtime after --idle-minutes without work (no compute units burn).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import signal
import string
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifacts  # noqa: E402

MAX_RESULT_FILE = 25 * 1024 * 1024  # results travel through the status dataset: keep them small
KEEP_JOBS = 40  # logs/results of this many most recent jobs stay in the status dataset


def log(msg: str) -> None:
    print(f"[worker {dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def now() -> str:
    return f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}"


def kaggle_bin() -> str:
    return shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")


def run_quiet(cmd: list[str], timeout: int = 600, **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, **kw)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, e.stdout or "", f"timeout after {timeout}s")
    except OSError as e:
        return subprocess.CompletedProcess(cmd, 127, "", f"{type(e).__name__}: {e}")


def default_env(root: str, user: str) -> dict:
    return {
        "LCDRU_TEXTS": f"{root}/texts",
        "LCDRU_RUNS": f"{root}/runs",
        "HF_HOME": f"{root}/hf",
        "KAGGLE_USERNAME": user,
        "PYTHONUNBUFFERED": "1",
    }


def pick_job(jobs: list[dict], worker: str, finished: dict, has_gpu: bool) -> dict | None:
    """First enabled job of this worker that is not finished, whose `after` jobs succeeded and whose GPU need fits."""
    for j in jobs:
        if str(j.get("worker", "A")).upper() != worker or j.get("disabled") or j["id"] in finished:
            continue
        if any(finished.get(d, {}).get("rc") != 0 for d in j.get("after", [])):
            continue
        if j.get("gpu") and not has_gpu:
            continue
        return j
    return None


class Worker:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.letter = a.worker.upper()
        self.k = kaggle_bin()
        self.home = os.path.join(a.root, "worker")
        self.status_dir = os.path.join(self.home, f"status_{self.letter.lower()}")
        self.status_id = f"{a.user}/lcd-ru-colab-status-{self.letter.lower()}"
        for d in ("logs", "results"):
            os.makedirs(os.path.join(self.status_dir, d), exist_ok=True)
        os.makedirs(os.path.join(self.home, "logs"), exist_ok=True)
        self.finished: dict[str, dict] = {}
        self.prev_status: dict = {}
        self.last_report = 0.0
        self.current: dict | None = None
        self.gpu_name = ""
        self.setup_tail = ""
        self.env = {**os.environ, **default_env(a.root, a.user), "LCDRU_SETUP_ARTIFACTS": a.setup_artifacts}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ kaggle transport
    def fetch_queue(self) -> dict:
        d = os.path.join(self.home, "queue")
        shutil.rmtree(d, ignore_errors=True)
        r = run_quiet(
            [self.k, "datasets", "download", f"{self.a.user}/{self.a.jobs_dataset}", "-p", d, "--unzip", "-o", "-q"]
        )
        files = glob.glob(os.path.join(d, "**", "jobs.json"), recursive=True)
        if r.returncode != 0 or not files:
            log(f"cannot read the jobs dataset ({(r.stderr or r.stdout)[-200:].strip()})")
            return {}
        with open(files[0]) as f:
            return json.load(f)

    def restore_history(self) -> None:
        """finished jobs + log tails + results of earlier sessions from the status dataset."""
        tmp = self.status_dir + "_prev"
        shutil.rmtree(tmp, ignore_errors=True)
        r = run_quiet([self.k, "datasets", "download", self.status_id, "-p", tmp, "--unzip", "-o", "-q"])
        p = os.path.join(tmp, "status.json")
        if r.returncode != 0 or not os.path.exists(p):
            log("no earlier status dataset: fresh history")
            return
        with open(p) as f:
            prev = json.load(f)
        self.finished = prev.get("finished", {})
        self.prev_status = {k: prev.get(k) for k in ("state", "time_utc", "gpu")}
        for d in ("logs", "results"):
            if os.path.isdir(os.path.join(tmp, d)):
                shutil.copytree(os.path.join(tmp, d), os.path.join(self.status_dir, d), dirs_exist_ok=True)
        log(f"history restored: {len(self.finished)} finished jobs")

    def refresh_code(self, want_version: str | None, wait_minutes: float = 15) -> bool:
        """Download the code dataset; if the job needs a newer version than Kaggle serves yet, retry for a while."""
        t0 = time.time()
        while True:
            tmp = self.a.code_dir + ".new"
            shutil.rmtree(tmp, ignore_errors=True)
            r = run_quiet(
                [self.k, "datasets", "download", f"{self.a.user}/lcd-ru-code", "-p", tmp, "--unzip", "-o", "-q"]
            )
            vfile = os.path.join(tmp, "CODE_VERSION")
            got = open(vfile).read().strip() if os.path.exists(vfile) else ""
            ok = r.returncode == 0 and os.path.exists(os.path.join(tmp, "src", "__init__.py"))
            if ok and (not want_version or got >= want_version):
                shutil.rmtree(self.a.code_dir, ignore_errors=True)
                os.replace(tmp, self.a.code_dir)
                log(f"code refreshed (version {got or '?'})")
                return True
            if time.time() - t0 > wait_minutes * 60:
                log(f"code {got or '?'} < wanted {want_version} after {wait_minutes:.0f} min; keeping the current copy")
                return False
            log(f"code dataset serves {got or '?'}, job wants {want_version}: waiting for Kaggle to process the upload")
            time.sleep(60)

    def report(self, state: str, tail: str = "") -> None:
        self.last_report = time.time()
        with self.lock:
            st = {
                "worker": self.letter,
                "time_utc": now(),
                "state": state,
                "current": dict(self.current) if self.current else None,
                "gpu": self.gpu_name,
                "finished": self.finished,
            }
            with open(os.path.join(self.status_dir, "status.json"), "w") as f:
                json.dump(st, f, indent=1)
            if tail:
                with open(os.path.join(self.status_dir, "session_tail.txt"), "w") as f:
                    f.write(tail[-20000:])
            meta = {"title": self.status_id.split("/")[1], "id": self.status_id, "licenses": [{"name": "CC0-1.0"}]}
            with open(os.path.join(self.status_dir, "dataset-metadata.json"), "w") as f:
                json.dump(meta, f)
            stage = self.status_dir + "_stage"  # upload a snapshot, the live dir keeps changing
            shutil.rmtree(stage, ignore_errors=True)
            shutil.copytree(self.status_dir, stage)
        r = run_quiet([self.k, "datasets", "version", "-p", stage, "-m", state, "--dir-mode", "zip", "-q"], timeout=600)
        if r.returncode != 0:
            r = run_quiet([self.k, "datasets", "create", "-p", stage, "--dir-mode", "zip", "-q"], timeout=600)
        if r.returncode != 0:
            log(f"status upload failed: {(r.stderr or r.stdout)[-200:].strip()}")

    # ------------------------------------------------------------------ jobs
    def expand(self, text: str) -> str:
        """$LCDRU_* from the job environment (not the worker's os.environ)."""
        return string.Template(text).safe_substitute(self.env)

    def collect_results(self, job: dict) -> list[str]:
        out = os.path.join(self.status_dir, "results", job["id"])
        shutil.rmtree(out, ignore_errors=True)
        got = []
        for pat in job.get("results", []):
            pat = self.expand(pat)
            for p in glob.glob(pat if os.path.isabs(pat) else os.path.join(self.a.code_dir, pat), recursive=True):
                if os.path.isfile(p) and os.path.getsize(p) <= MAX_RESULT_FILE:
                    os.makedirs(out, exist_ok=True)
                    dst = os.path.join(out, os.path.basename(os.path.dirname(p)) + "__" + os.path.basename(p))
                    shutil.copy2(p, dst)
                    got.append(os.path.basename(dst))
        return got

    def save_artifacts(self, names: list[str], env: dict, message: str) -> list[str]:
        msgs = []
        for name in names:
            msg = artifacts.save(name, env, message=message)
            log(msg)
            msgs.append(msg)
        return msgs

    def prune(self) -> None:
        order = sorted(self.finished, key=lambda k: self.finished[k].get("end", ""))
        for old in order[:-KEEP_JOBS]:
            shutil.rmtree(os.path.join(self.status_dir, "results", old), ignore_errors=True)
            p = os.path.join(self.status_dir, "logs", f"{old}.log")
            if os.path.exists(p):
                os.remove(p)

    def run_job(self, job: dict) -> int:
        env = {**self.env, **{k: self.expand(str(v)) for k, v in job.get("env", {}).items()}}
        full_log = os.path.join(self.home, "logs", f"{job['id']}.log")
        tail_log = os.path.join(self.status_dir, "logs", f"{job['id']}.log")
        for name in job.get("restore", []):
            log(artifacts.restore(name, env))
        t0 = time.time()
        with self.lock:
            self.current = {"id": job["id"], "cmd": job["cmd"], "start": now(), "pid": None}
        log(f"START {job['id']}: {job['cmd']}")
        self.report("running")
        tail: list[str] = []
        stop = threading.Event()
        reason = {"why": ""}
        run_saves = [n for n in job.get("save", []) if n.startswith("runs/")]

        def watchdog(p: subprocess.Popen) -> None:
            last_report = last_save = time.time()
            while not stop.wait(30):
                if job.get("timeout_h") and time.time() - t0 > float(job["timeout_h"]) * 3600:
                    reason["why"] = f"timeout after {job['timeout_h']} h"
                elif job.get("save_every") and run_saves and time.time() - last_save > float(job["save_every"]) * 60:
                    last_save = time.time()
                    self.save_artifacts(run_saves, env, f"{job['id']} checkpoint")
                elif time.time() - last_report > self.a.report_minutes * 60:
                    last_report = time.time()
                    if job["id"] in self.fetch_queue().get("cancel", []):
                        reason["why"] = "cancelled from the queue"
                    else:
                        with open(tail_log, "w") as f:
                            f.write("".join(tail[-self.a.tail_lines :]))
                        with self.lock:
                            self.current["elapsed_min"] = round((time.time() - t0) / 60, 1)
                        self.report("running")
                if reason["why"]:
                    log(f"killing {job['id']}: {reason['why']}")
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                        for _ in range(20):  # trainers save on SIGTERM; give them a moment
                            if p.poll() is not None:
                                break
                            time.sleep(1)
                        else:
                            os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return

        with open(full_log, "w") as flog, subprocess.Popen(
            ["bash", "-c", job["cmd"]], cwd=self.a.code_dir, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        ) as p:  # fmt: skip
            with self.lock:
                self.current["pid"] = p.pid
            threading.Thread(target=watchdog, args=(p,), daemon=True).start()
            for line in p.stdout:
                sys.stdout.write(line)
                flog.write(line)
                flog.flush()
                tail.append(line)
                del tail[: -self.a.keep_lines]
            rc = p.wait()
        stop.set()
        if reason["why"]:
            tail.append(f"\n[worker] {reason['why']}\n")
            rc = rc if rc not in (0, None) else 143
        with open(tail_log, "w") as f:
            f.write("".join(tail))
        results = self.collect_results(job)
        to_save = [n for n in job.get("save", []) if rc == 0 or n.startswith("runs/")]
        saves = self.save_artifacts(to_save, env, f"{job['id']} rc={rc}")
        minutes = round((time.time() - t0) / 60, 1)
        with self.lock:
            self.finished[job["id"]] = {
                "rc": rc, "end": now(), "minutes": minutes, "results": results, "saves": saves,
                "note": reason["why"] + ("; SAVE FAILED" if any("FAILED" in s for s in saves) else ""),
            }  # fmt: skip
            self.current = None
        self.prune()
        log(f"{'DONE' if rc == 0 else 'FAILED'} {job['id']} (exit code {rc}, {minutes} min)")
        self.report("job_done" if rc == 0 else "job_failed")
        return rc

    # ------------------------------------------------------------------ session
    def other_worker_alive(self) -> bool:
        """Two workers with the same letter would run the same jobs and overwrite each other's status. A live worker
        reports at least every --report-minutes (also while idle), so a recent non-final status means it is alive."""
        st, t = self.prev_status.get("state"), self.prev_status.get("time_utc")
        if self.a.force or not t or st in ("stopped", "setup_failed", "other_worker_alive"):
            return False
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=dt.timezone.utc)).total_seconds() / 60  # fmt: skip
        if age > self.a.report_minutes + 5:
            return False
        log(f"STOP: worker {self.letter} reported '{st}' {age:.0f} min ago on {self.prev_status.get('gpu') or '?'}: "
            f"another session is probably running it. Close the other tab, or start with --force (or another --worker "
            f"letter) if it is really dead.")  # fmt: skip
        return True

    def gpu(self) -> str:
        r = run_quiet(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=60)
        return r.stdout.strip() if r.returncode == 0 else ""

    def setup(self) -> int:
        script = os.path.join(self.a.code_dir, "colab", "session_setup.sh")
        log("session setup (pip, data artifact from Kaggle) - first time ~3-5 min")
        r = run_quiet(["bash", script], timeout=3 * 3600, cwd=self.a.code_dir, env=self.env)
        out = (r.stdout or "") + (r.stderr or "")
        with open(os.path.join(self.home, "logs", f"setup_{self.letter.lower()}.log"), "w") as f:
            f.write(out)
        print(out[-4000:], flush=True)
        self.setup_tail = out[-20000:]
        return r.returncode

    def loop(self) -> None:
        t0, last_work = time.time(), time.time()
        self.gpu_name = self.gpu()
        log(f"worker {self.letter} on {self.gpu_name or 'CPU only'}; queue {self.a.user}/{self.a.jobs_dataset}; "
            f"idle limit {self.a.idle_minutes:.0f} min")  # fmt: skip
        self.restore_history()
        if self.other_worker_alive():
            self.release()  # do not burn compute units on a runtime that must not work
            return
        if not self.a.skip_setup:
            self.refresh_code(None)
        rc = 0 if self.a.skip_setup else self.setup()
        self.report("ready" if rc == 0 else "setup_failed", tail=self.setup_tail)
        if rc != 0:
            log("session setup failed (see the tail above); stopping")
            return
        while (time.time() - t0) / 3600 < self.a.max_hours:
            queue = self.fetch_queue()
            if self.letter in [str(x).upper() for x in queue.get("stop_workers", [])]:
                log(f"the queue asks worker {self.letter} to stop")
                break
            job = pick_job(queue.get("jobs", []), self.letter, self.finished, bool(self.gpu_name))
            if job is None:
                idle = (time.time() - last_work) / 60
                if idle >= self.a.idle_minutes or self.a.once:
                    log(f"no jobs for worker {self.letter} for {idle:.0f} min: stopping")
                    break
                log(f"no runnable jobs (idle {idle:.0f} min), next check in {self.a.poll_minutes:.0f} min")
                if time.time() - self.last_report > self.a.report_minutes * 60:  # heartbeat while idle
                    self.report("idle")
                time.sleep(self.a.poll_minutes * 60)
                continue
            self.refresh_code(job.get("code_version"))
            if job.get("setup_again"):
                self.setup()
            self.run_job(job)
            last_work = time.time()
        self.report("stopped")
        self.release()

    def release(self) -> None:
        if self.a.no_unassign:
            return
        log("releasing the runtime")
        try:
            from google.colab import runtime

            runtime.unassign()
        except Exception as e:  # noqa: BLE001  (not on Colab)
            log(f"unassign skipped: {e}")


def worker_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="A", help="letter of this session; it runs only jobs with the same letter")
    ap.add_argument("--user", default="kirillkholkin")
    ap.add_argument("--jobs-dataset", default="lcd-ru-jobs")
    ap.add_argument("--root", default="/content", help="local working root: texts/, runs/, hf/")
    ap.add_argument("--code-dir", default="/content/code")
    ap.add_argument("--idle-minutes", type=float, default=45)
    ap.add_argument("--poll-minutes", type=float, default=2)
    ap.add_argument("--report-minutes", type=float, default=10, help="status upload period while a job runs")
    ap.add_argument("--tail-lines", type=int, default=400, help="log lines published while a job runs")
    ap.add_argument("--keep-lines", type=int, default=2000, help="log lines published when a job ends")
    ap.add_argument("--max-hours", type=float, default=23)
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument("--setup-artifacts", default="data", help="artifacts restored at session start")
    ap.add_argument("--no-unassign", action="store_true")
    ap.add_argument("--once", action="store_true", help="stop as soon as there is nothing to do (tests)")
    ap.add_argument("--force", action="store_true", help="start even if another worker with this letter looks alive")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    Worker(worker_args(argv)).loop()


if __name__ == "__main__":
    main()
