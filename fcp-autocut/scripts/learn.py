#!/usr/bin/env python3
"""내 유튜브 영상(완성본)으로 '내 편집 습관'과 '내 콘텐츠'를 배우기.

사용:
  learn.py add <유튜브 링크> --format vlog [--video]   # 영상 하나 분석해서 저장
  learn.py summary                                    # 지금까지 배운 것 요약 + 설정값 제안
  learn.py list                                       # 분석한 영상 목록

- 받은 오디오/영상은 임시 폴더에서만 쓰고 바로 지움. 저장되는 건 숫자 통계와 대본 텍스트뿐.
- 완성본의 '문장 사이 멈춤 길이'는 내가 실제로 얼마나 촘촘하게 자르는지 보여 줌 → presets 값 제안에 사용.
- 제안만 하고 presets.yaml은 바꾸지 않음(사용자 동의 후 Claude가 수정).
- 화면 내용은 보지 않음. --video를 주면 화면 전환(컷) 횟수만 셈.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (FORMAT_KO, FORMATS, data_home, die, fmt_time, has_hangul, load_json, load_presets,
                    load_sectioned, load_wordlist, norm, save_json)


def channel_dir() -> Path:
    return data_home() / "channel"


def ytdlp() -> list[str]:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        die("yt-dlp가 없어요. setup.sh --youtube 로 설치해 주세요.")


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        die(f"유튜브에서 가져오지 못했어요.\n{(r.stderr or r.stdout)[-600:]}\n"
            "(비공개·멤버십 영상이거나, 유튜브가 잠시 막았을 수 있어요. yt-dlp 업데이트: setup.sh --youtube)")
    return r


# ───────── 자막(VTT) ─────────

def parse_vtt(text: str) -> list[str]:
    """VTT 자막 → 문장 목록(시간·태그 제거, 연속 중복 제거)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit() or \
                line.startswith(("Kind:", "Language:", "NOTE", "STYLE")):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and (not out or out[-1] != line):
            out.append(line)
    return out


# ───────── 통계 ─────────

def compute_stats(segments: list[dict], duration: float, cfg: dict) -> dict:
    """완성본 대본 → 멈춤 길이·말 속도·남아 있는 군말·자주 쓰는 말."""
    words = [w for s in segments for w in s["words"]]
    words.sort(key=lambda w: w["s"])
    split = cfg["sentence"]["split_pause_sec"]
    in_gaps, between = [], []
    for a, b in zip(words, words[1:]):
        g = b["s"] - a["e"]
        if g < 0:
            continue
        if re.search(r"[\.\?\!]$", a["w"]) or g >= split:
            between.append(g)
        else:
            in_gaps.append(g)
    speak = sum(max(0.0, w["e"] - w["s"]) for w in words)
    syl = sum(sum(1 for ch in w["w"] if "가" <= ch <= "힣") for w in words)
    fillers = load_sectioned("fillers.txt")
    strong = {norm(x) for x in fillers.get("확실", [])}
    weak = {norm(x) for x in fillers.get("애매", [])}
    nw = [norm(w["w"]) for w in words]
    minutes = max(duration / 60, 1e-9)
    stop = strong | weak | {norm(x) for x in ["그리고", "그래서", "이거", "저는", "제가", "진짜", "너무", "있는", "하는",
                                              "이렇게", "근데", "이런", "그런", "여기", "정말", "되게", "많이", "같아요"]}
    freq = Counter(x for x in nw if len(x) >= 2 and x not in stop and has_hangul(x))

    def pct(a, q):
        return round(float(np.percentile(a, q)), 3) if len(a) else None

    text = " ".join(w["w"] for w in words)
    first = " ".join(w["w"] for w in words if w["s"] <= 30)
    last = " ".join(w["w"] for w in words if w["s"] >= duration - 30)
    return {
        "duration": round(duration, 1),
        "speech_ratio": round(speak / max(duration, 1e-9), 3),
        "syllables_per_sec": round(syl / max(speak, 1e-9), 2),
        "pause_between_sentences": {"median": pct(between, 50), "p25": pct(between, 25), "p75": pct(between, 75),
                                    "p90": pct(between, 90), "n": len(between)},
        "pause_inside_sentence": {"median": pct(in_gaps, 50), "p90": pct(in_gaps, 90), "n": len(in_gaps)},
        "fillers_per_min": {"strong": round(sum(x in strong for x in nw) / minutes, 2),
                            "weak": round(sum(x in weak for x in nw) / minutes, 2)},
        "kept_fillers": dict(Counter(x for x in nw if x in strong or x in weak).most_common(10)),
        "top_words": dict(freq.most_common(25)),
        "opening_30s": first,
        "closing_30s": last,
        "text": text,
    }


def suggest_presets(videos: list[dict]) -> dict:
    """완성본의 문장 사이 멈춤으로 포맷별 무음 여유값 제안."""
    out = {}
    for fmt in FORMATS:
        vs = [v for v in videos if v.get("format") == fmt and v["stats"]["pause_between_sentences"]["median"]]
        if not vs:
            continue
        med = float(np.median([v["stats"]["pause_between_sentences"]["median"] for v in vs]))
        cur = load_presets(fmt)["silence"]
        cur_total = cur["pad_before_sec"] + cur["pad_after_sec"]
        # 완성본에서 남긴 멈춤 ≈ 컷 뒤 남는 여유(pad_after + pad_before)
        target = max(0.08, med)
        ratio = cur["pad_after_sec"] / max(cur_total, 1e-9)
        out[fmt] = {"videos": len(vs), "final_pause_median": round(med, 3),
                    "current_pad_total": round(cur_total, 3),
                    "suggest_pad_after_sec": round(target * ratio, 3),
                    "suggest_pad_before_sec": round(target * (1 - ratio), 3),
                    "note": ("현재 설정이 완성본보다 멈춤을 더 길게 남겨요 → 조금 더 촘촘히" if cur_total > target * 1.25 else
                             "현재 설정이 완성본보다 촘촘해요 → 조금 여유 있게" if cur_total < target * 0.8 else
                             "현재 설정이 완성본과 비슷해요")}
    return out


# ───────── 명령 ─────────

def cmd_add(args):
    cfg = load_presets(args.format)
    meta = json.loads(run(ytdlp() + ["-J", "--skip-download", "--no-playlist", args.url]).stdout)
    vid = meta["id"]
    out = channel_dir() / "videos" / f"{vid}.json"
    if out.exists() and not args.force:
        print(f"이미 분석한 영상이에요: {meta.get('title')} (다시 하려면 --force)")
        return
    from transcribe import EXIT_NEED_MODEL, build_prompt, check_platform, chunk_bounds, model_cached, run_whisper, simplify
    check_platform()
    if not model_cached(cfg["transcribe"]["model"]) and not args.allow_download:
        print(f"NEED_MODEL_DOWNLOAD: 받아쓰기 모델({cfg['transcribe']['model_size_note']})이 아직 없어요.")
        sys.exit(EXIT_NEED_MODEL)
    from common import ffmpeg_pcm
    with tempfile.TemporaryDirectory(prefix="fcpautocut_yt_") as td:
        td = Path(td)
        run(ytdlp() + ["-f", cfg["youtube"]["audio_format"], "--no-playlist", "-o", str(td / "audio.%(ext)s"), args.url])
        audio_file = next(td.glob("audio.*"))
        subs = []
        if meta.get("subtitles"):
            subprocess.run(ytdlp() + ["--skip-download", "--write-subs", "--sub-langs", cfg["youtube"]["sub_langs"],
                                      "--sub-format", "vtt", "--no-playlist", "-o", str(td / "sub.%(ext)s"), args.url],
                           capture_output=True)
            for f in td.glob("sub*.vtt"):
                subs += parse_vtt(f.read_text(encoding="utf-8", errors="ignore"))
        sr = cfg["analysis"]["sample_rate"]
        audio = ffmpeg_pcm(audio_file, sr, 1)[:, 0].copy()
        duration = len(audio) / sr
        prompt = build_prompt(cfg)
        segs = []
        for i, (a, b) in enumerate(chunk_bounds(audio, sr, cfg), 1):
            segs += simplify(run_whisper(audio[a:b], cfg, prompt), a / sr)
            print(f"받아쓰기 {fmt_time(b / sr)} / {fmt_time(duration)}", flush=True)
        scenes = None
        if args.video:
            run(ytdlp() + ["-f", "worstvideo[height>=240]/worst", "--no-playlist", "-o", str(td / "video.%(ext)s"), args.url])
            vf = next(td.glob("video.*"))
            thr = cfg["youtube"]["scene_threshold"]
            r = subprocess.run(["ffmpeg", "-nostdin", "-i", str(vf), "-vf", f"select='gt(scene,{thr})',showinfo",
                                "-f", "null", "-"], capture_output=True, text=True)
            n = len(re.findall(r"pts_time:", r.stderr))
            scenes = {"cuts": n, "avg_shot_sec": round(duration / (n + 1), 2)}
    stats = compute_stats(segs, duration, cfg)
    vocab = {norm(x) for x in load_wordlist("vocab.txt")}
    sub_words = Counter(w for line in subs for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", line))
    record = {"id": vid, "url": args.url, "title": meta.get("title"), "upload_date": meta.get("upload_date"),
              "format": args.format, "analyzed": datetime.now().isoformat(timespec="seconds"),
              "view_count": meta.get("view_count"), "like_count": meta.get("like_count"),
              "chapters": [c.get("title") for c in meta.get("chapters") or []],
              "description": (meta.get("description") or "")[:1500],
              "has_manual_subs": bool(subs), "scenes": scenes, "stats": stats,
              "subtitle_text": "\n".join(subs)[:20000],
              "vocab_hints": [w for w, n in sub_words.most_common(60) if n >= 2 and norm(w) not in vocab][:30]}
    save_json(out, record)
    s = stats
    pb = s["pause_between_sentences"]
    print(f"✅ 분석 저장: {meta.get('title')} ({FORMAT_KO[args.format]}, {fmt_time(duration)})")
    print(f"  • 말하는 비율 {s['speech_ratio']:.0%}, 말 속도 초당 {s['syllables_per_sec']}음절")
    print(f"  • 문장 사이 멈춤(완성본): 중간값 {pb['median']}초 / 긴 편(90%) {pb['p90']}초")
    print(f"  • 남겨 둔 군말: 분당 확실 {s['fillers_per_min']['strong']}회, 애매 {s['fillers_per_min']['weak']}회")
    if scenes:
        print(f"  • 화면 전환 {scenes['cuts']}회, 평균 {scenes['avg_shot_sec']}초마다")
    if record["vocab_hints"]:
        print(f"  • 직접 올린 자막에 자주 나온 단어(고유명사 사전 후보): {', '.join(record['vocab_hints'][:15])}")
    print(f"VIDEO_JSON={out}")


def load_videos():
    d = channel_dir() / "videos"
    return [load_json(f) for f in sorted(d.glob("*.json"))] if d.exists() else []


def cmd_summary(args):
    vs = load_videos()
    if not vs:
        print("아직 분석한 영상이 없어요. learn.py add <링크> 로 추가해 주세요.")
        return
    print(f"📚 분석한 영상 {len(vs)}개")
    for fmt in FORMATS:
        sub = [v for v in vs if v.get("format") == fmt]
        if not sub:
            continue
        S = [v["stats"] for v in sub]
        med = lambda xs: round(float(np.median([x for x in xs if x is not None])), 3) if any(x is not None for x in xs) else None
        print(f"\n[{FORMAT_KO[fmt]}] {len(sub)}개")
        print(f"  • 문장 사이 멈춤 중간값: {med([s['pause_between_sentences']['median'] for s in S])}초")
        print(f"  • 말 속도: 초당 {med([s['syllables_per_sec'] for s in S])}음절, 말하는 비율 {med([s['speech_ratio'] for s in S])}")
        print(f"  • 남겨 둔 군말(분당): 확실 {med([s['fillers_per_min']['strong'] for s in S])}, "
              f"애매 {med([s['fillers_per_min']['weak'] for s in S])}")
        sc = [v["scenes"]["avg_shot_sec"] for v in sub if v.get("scenes")]
        if sc:
            print(f"  • 평균 화면 전환 간격: {round(float(np.median(sc)), 2)}초")
    sug = suggest_presets(vs)
    if sug:
        print("\n🔧 설정값 제안(presets.yaml은 아직 안 바꿈):")
        for fmt, x in sug.items():
            print(f"  [{FORMAT_KO[fmt]}] 완성본 멈춤 {x['final_pause_median']}초 vs 현재 남기는 여유 {x['current_pad_total']}초 "
                  f"→ pad_after {x['suggest_pad_after_sec']} / pad_before {x['suggest_pad_before_sec']} ({x['note']})")
    hints = Counter(w for v in vs for w in v.get("vocab_hints", []))
    if hints:
        print("\n📖 고유명사 사전 후보: " + ", ".join(w for w, _ in hints.most_common(20)))
    notes = channel_dir() / "채널노트.md"
    print(f"\n채널 노트: {notes} ({'있음' if notes.exists() else '아직 없음 — Claude가 대본을 읽고 작성'})")
    save_json(channel_dir() / "summary.json", {"videos": len(vs), "suggest": sug, "vocab_hints": dict(hints)})


def cmd_list(args):
    for v in load_videos():
        print(f"- {v.get('upload_date', '')} [{FORMAT_KO.get(v.get('format'), '?')}] {v.get('title')}  "
              f"({fmt_time(v['stats']['duration'])}) {channel_dir() / 'videos' / (v['id'] + '.json')}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("add")
    a.add_argument("url")
    a.add_argument("--format", required=True, choices=FORMATS)
    a.add_argument("--video", action="store_true", help="화면 전환 횟수도 셈(저화질 영상을 잠깐 받음)")
    a.add_argument("--allow-download", action="store_true")
    a.add_argument("--force", action="store_true")
    sp.add_parser("summary")
    sp.add_parser("list")
    args = ap.parse_args()
    {"add": cmd_add, "summary": cmd_summary, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    main()
