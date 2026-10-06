"""fcp-autocut 공통 도구: 설정 읽기, 경로, 오디오 분석, 프레임 계산."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

SKILL_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = SKILL_DIR / "config"
FORMATS = ("narration", "vlog", "interview")
FORMAT_KO = {"narration": "나레이션", "vlog": "브이로그", "interview": "인터뷰"}


def data_home() -> Path:
    """작업 파일 보관 위치. 원본 폴더에는 아무것도 쓰지 않기 위해 따로 둔다."""
    env = os.environ.get("FCP_AUTOCUT_HOME")
    return Path(env) if env else Path.home() / ".local/share/fcp-autocut"


def die(msg: str, code: int = 1):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(code)


# ───────── 설정 ─────────

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_presets(fmt: str | None = None, path: Path | None = None) -> dict:
    raw = yaml.safe_load((path or CONFIG_DIR / "presets.yaml").read_text(encoding="utf-8"))
    common = raw["common"]
    if fmt is None:
        return common
    if fmt not in FORMATS:
        die(f"포맷은 {', '.join(FORMATS)} 중 하나여야 해요: {fmt}")
    return _deep_merge(common, raw[fmt])


def load_wordlist(name: str) -> list[str]:
    p = CONFIG_DIR / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("["):
            out.append(line)
    return out


def load_sectioned(name: str) -> dict[str, list[str]]:
    """[섹션] 헤더로 나뉜 사전 파일."""
    sections: dict[str, list[str]] = {}
    cur = "_"
    for line in (CONFIG_DIR / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[(.+)\]", line)
        if m:
            cur = m.group(1)
            continue
        sections.setdefault(cur, []).append(line)
    return sections


# ───────── 텍스트 ─────────

_PUNCT = re.compile(r"[\s\.\,\?\!\…\·\~\-\"\'\(\)\[\]「」『』“”‘’:;]+")


def norm(text: str) -> str:
    """비교용: 유니코드 정리 + 띄어쓰기·문장부호 제거 + 소문자."""
    text = unicodedata.normalize("NFC", text or "")
    return _PUNCT.sub("", text).lower()


def has_hangul(text: str) -> bool:
    return any("가" <= c <= "힣" or "ㄱ" <= c <= "ㆎ" for c in text)


# ───────── 프로젝트 / 배치 ─────────

def project_dir(title: str) -> Path:
    safe = re.sub(r"[/\\:]", "_", title).strip() or "untitled"
    return data_home() / "projects" / safe


def resolve_path(s: str) -> Path:
    """준 경로 그대로 쓰되, 한글 자모 조합 방식(NFC/NFD) 차이로 못 찾으면 다른 방식으로 찾아본다."""
    for cand in (s, unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s)):
        if Path(cand).exists():
            return Path(cand)
    return Path(s)


def read_batch(pdir: Path) -> list[Path]:
    """batch.txt: 사용자가 준 정확한 경로 목록(한 줄에 하나). glob 금지."""
    f = pdir / "batch.txt"
    if not f.exists():
        die(f"batch.txt가 없어요: {f}")
    paths = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            paths.append(resolve_path(line.rstrip("\r")))
    return paths


def clip_ids(paths: list[Path]) -> list[str]:
    """파일 이름(확장자 제외)을 이름표로. 같은 이름이 있으면 _2, _3을 붙인다."""
    seen: dict[str, int] = {}
    ids = []
    for p in paths:
        stem = unicodedata.normalize("NFC", p.stem)
        n = seen.get(stem, 0) + 1
        seen[stem] = n
        ids.append(stem if n == 1 else f"{stem}_{n}")
    return ids


def file_signature(p: Path) -> dict:
    st = p.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def load_json(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def fmt_time(t: float) -> str:
    t = round(max(0.0, float(t)), 1)   # 반올림을 먼저 해야 59.96초가 "60.0초"로 보이지 않음
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


# ───────── 오디오 ─────────

def ffmpeg_pcm(src: Path, sr: int, channels: int, stream_index: int = 0,
               start: float | None = None, dur: float | None = None) -> np.ndarray:
    """원본을 건드리지 않고 메모리로만 PCM을 읽는다. 반환: (샘플, 채널) float32."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-map", f"0:a:{stream_index}", "-vn", "-ac", str(channels), "-ar", str(sr),
            "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        die(f"오디오를 읽지 못했어요: {src.name}\n{r.stderr.decode(errors='ignore')[-400:]}")
    a = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return a.reshape(-1, channels)


def rms_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    fr = x[: n * hop].reshape(n, hop)
    r = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
    return (20 * np.log10(r + 1e-9)).astype(np.float32)


def envelope(src: Path, pdir: Path, cid: str, cfg: dict, stereo: bool = False, pcm: np.ndarray | None = None) -> dict:
    """소리 크기 곡선(10ms 단위 dB). 작업 폴더에 캐시 → 같은 영상은 한 번만 디코딩.
    pcm을 주면(probe가 이미 읽은 소리) 디코딩 없이 캐시를 채운다."""
    a = cfg["analysis"]
    sr, hop_sec = a["sample_rate"], a["hop_sec"]
    hop = int(round(sr * hop_sec))
    sig = file_signature(src)
    key = hashlib.md5(f"{src}|{sig}|{sr}|{hop}|{stereo}".encode()).hexdigest()[:12]
    cache = pdir / "cache" / f"{cid}.{key}.npz"
    if cache.exists() and pcm is None:
        z = np.load(cache)
        return {k: z[k] for k in z.files} | {"hop_sec": hop_sec}
    out = {}
    if stereo:
        if pcm is None:
            pcm = ffmpeg_pcm(src, sr, 2)
        out["l"] = rms_db(pcm[:, 0], hop)
        out["r"] = rms_db(pcm[:, 1], hop)
        out["mono"] = rms_db(pcm.mean(axis=1), hop)
    else:
        out["mono"] = rms_db((pcm if pcm is not None else ffmpeg_pcm(src, sr, 1))[:, 0], hop)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out | {"hop_sec": hop_sec}


def clip_envelope(clip: dict, pdir: Path, cfg: dict) -> tuple[np.ndarray | None, dict | None]:
    """분석·컷 보정에 쓸 소리 곡선(한쪽만 녹음됐으면 그쪽 채널). 반환: (곡선, 전체 dict)"""
    if not clip.get("audio"):
        return None, None
    stereo = clip["audio"]["channels"] >= 2
    e = envelope(Path(clip["path"]), pdir, clip["id"], cfg, stereo=stereo)
    key = {"left": "l", "right": "r"}.get(clip["channel"].get("use_channel") or "", "mono")
    return e.get(key, e["mono"]), e


def levels(env: np.ndarray, cfg: dict) -> dict:
    a = cfg["analysis"]
    if len(env) == 0:
        return {"noise_db": -90.0, "speech_db": -90.0, "range_db": 0.0}
    noise = float(np.percentile(env, a["noise_percentile"]))
    speech = float(np.percentile(env, a["speech_percentile"]))
    return {"noise_db": noise, "speech_db": speech, "range_db": speech - noise}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 연속 구간 [시작, 끝) 인덱스 목록."""
    if len(mask) == 0:
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


# ───────── 프레임 / 시간 ─────────

def parse_rate(s: str) -> Fraction:
    """ffprobe 'r_frame_rate'(예: 30000/1001) → 초당 프레임 수."""
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    return Fraction(s)


def frame_duration(fps: Fraction) -> Fraction:
    """29.97 → 1001/30000 처럼 정확한 분수로."""
    fd = 1 / fps
    # 23.976/29.97/59.94 등은 정확히 1001 계열로 맞춘다.
    for nominal in (24, 30, 60, 48, 120):
        if abs(float(fps) - nominal * 1000 / 1001) < 0.005:
            return Fraction(1001, nominal * 1000)
    for nominal in (24, 25, 30, 50, 60, 48, 100, 120):
        if abs(float(fps) - nominal) < 0.005:
            return Fraction(1, nominal)
    return fd


def to_frames_floor(t: float | Fraction, fd: Fraction) -> int:
    return int(Fraction(t).limit_denominator(10**9) // fd)


def to_frames_round(t: float | Fraction, fd: Fraction) -> int:
    return int(round(Fraction(t).limit_denominator(10**9) / fd))


def to_frames_ceil(t: float | Fraction, fd: Fraction) -> int:
    q = Fraction(t).limit_denominator(10**9) / fd
    return int(-(-q.numerator // q.denominator))


def ftime(frames: int, fd: Fraction) -> str:
    """프레임 수 → FCPXML 시간 문자열(예: 3003/30000s)."""
    if frames == 0:
        return "0s"
    v = frames * fd.numerator
    if v % fd.denominator == 0:
        return f"{v // fd.denominator}s"
    return f"{v}/{fd.denominator}s"


def timecode_to_frames(tc: str, fps: Fraction) -> int:
    """'01:00:00:00' 또는 드롭프레임 '01:00:00;00' → 프레임 수."""
    drop = ";" in tc
    parts = [int(x) for x in re.split(r"[:;.]", tc)]
    if len(parts) != 4:
        raise ValueError(tc)
    h, m, s, f = parts
    nominal = int(round(float(fps)))
    total_min = h * 60 + m
    frames = (total_min * 60 + s) * nominal + f
    if drop:
        dropn = 2 if nominal == 30 else 4 if nominal == 60 else 0
        frames -= dropn * (total_min - total_min // 10)
    return frames
