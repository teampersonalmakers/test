#!/usr/bin/env python3
"""준비 단계를 한 번에: 영상 확인 → 받아쓰기 → 후보 계산 → 초안.

사용: autocut.py --title 제목 --format vlog [--init 경로...] [--sort-name] [--allow-download]
- 받아쓰기 모델이 없으면 종료 코드 3(사용자 동의 후 --allow-download 로 다시).
- 이미 끝난 단계(받아쓰기 등)는 건너뛰므로 몇 번을 다시 실행해도 됨.
- 끝나면 Claude가 view.py로 대본을 읽고 edits.json / notes.json 작성 → build.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def step(name, *args):
    print(f"\n▶ {name}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / name), *args])
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--format", required=True)
    ap.add_argument("--init", nargs="+")
    ap.add_argument("--sort-name", action="store_true")
    ap.add_argument("--sort-time", action="store_true")
    ap.add_argument("--out-dir")
    ap.add_argument("--allow-download", action="store_true")
    a = ap.parse_args()
    t = ["--title", a.title]
    probe = [*t, "--format", a.format]
    if a.init:
        probe += ["--init", *a.init]
    if a.sort_name:
        probe.append("--sort-name")
    if a.sort_time:
        probe.append("--sort-time")
    if a.out_dir:
        probe += ["--out-dir", a.out_dir]
    step("probe.py", *probe)
    step("transcribe.py", *t, *(["--allow-download"] if a.allow_download else []))
    step("analyze.py", *t)
    step("view.py", *t, "--suggest")
    print("\nPREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
