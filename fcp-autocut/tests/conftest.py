"""테스트 공용 도구: 합성 영상(사인파+무음) 만들기, 가짜 받아쓰기, 스크립트 실행."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))

FCP_DTD_DIR = Path("/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources")


def find_test_dtd() -> Path | None:
    """맥: 파이널컷 앱 안의 최신 DTD. 그 외: FCP_AUTOCUT_TEST_DTD 환경변수."""
    env = os.environ.get("FCP_AUTOCUT_TEST_DTD")
    if env and Path(env).exists():
        return Path(env)
    if FCP_DTD_DIR.exists():
        import re
        c = []
        for f in FCP_DTD_DIR.glob("FCPXMLv1_*.dtd"):
            m = re.fullmatch(r"FCPXMLv(\d+)_(\d+)\.dtd", f.name)
            if m:
                c.append(((int(m.group(1)), int(m.group(2))), f))
        if c:
            return max(c)[1]
    return None


def tone_expr(intervals, amp=0.5, freq=220, noise=0.002):
    """intervals 안에서만 사인파, 나머지는 아주 작은 잡음."""
    if intervals:
        gate = "+".join(f"between(t,{a},{b})" for a, b in intervals)
        tone = f"{amp}*sin(2*PI*{freq}*t)*min(1,{gate})"
    else:
        tone = "0"
    return f"{tone}+{noise}*(random(0)-0.5)"


def make_video(path: Path, dur: float, left: list, right: list | None = None, rate="30000/1001",
               size="320x180", timecode: str | None = None, amp_r=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    el = tone_expr(left)
    er = tone_expr(right if right is not None else left, amp=amp_r)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
           "-f", "lavfi", "-i", f"aevalsrc='{el}|{er}':s=48000", "-t", str(dur),
           "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest"]
    if timecode:
        cmd += ["-timecode", timecode]
    cmd.append(str(path))
    subprocess.run(cmd, check=True)
    return path


def W(w, s, e, p=0.9):
    return {"w": w, "s": s, "e": e, "p": p}


def seg(*words):
    return {"start": words[0]["s"], "end": words[-1]["e"], "text": " ".join(w["w"] for w in words),
            "words": list(words)}


class Project:
    def __init__(self, home: Path, title: str):
        self.home, self.title = home, title
        self.env = dict(os.environ, FCP_AUTOCUT_HOME=str(home))
        self.dir = home / "projects" / title

    def run(self, script, *args, check=True):
        r = subprocess.run([sys.executable, str(SCRIPTS / script), "--title", self.title, *args],
                           env=self.env, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise AssertionError(f"{script} 실패:\n{r.stdout}\n{r.stderr}")
        return r

    def probe(self, fmt, paths):
        return self.run("probe.py", "--format", fmt, "--init", *[str(p) for p in paths])

    def write_transcript(self, clip_index: int, segments):
        probe = json.loads((self.dir / "probe.json").read_text(encoding="utf-8"))
        c = probe["clips"][clip_index]
        out = self.dir / "transcripts" / f"{c['id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"clip": c["id"], "path": c["path"], "signature": c["signature"], "model": "fake",
                                   "duration": c["duration"], "channel_used": "mix", "segments": segments},
                                  ensure_ascii=False), encoding="utf-8")
        return c["id"]

    def json(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def write(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def proj(tmp_path):
    def make(title="테스트 프로젝트"):
        return Project(tmp_path / "home", title)
    return make


@pytest.fixture
def dtd():
    d = find_test_dtd()
    if d is None:
        pytest.skip("FCPXML DTD 없음(맥에서는 파이널컷 앱에서 자동으로 찾음)")
    return d
