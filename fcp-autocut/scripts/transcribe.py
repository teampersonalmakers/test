#!/usr/bin/env python3
"""받아쓰기 (mlx-whisper, 단어별 시간·신뢰도).

사용: transcribe.py --title 제목 [--allow-download]
- 이미 받아쓴 영상은 건너뜀. 긴 영상은 조용한 지점에서 나눠서 조각마다 저장 → 끊겨도 이어서.
- 오디오는 메모리에서만 다루고 파일로 저장하지 않음(원본·변환 파일 모두 생성 안 함).
- 모델이 컴퓨터에 없으면 받지 않고 멈춤 → 사용자 동의 후 --allow-download 로 다시 실행.
"""
from __future__ import annotations

import argparse
import inspect
import platform
import sys
import time
from pathlib import Path

import numpy as np

from common import (die, ffmpeg_pcm, fmt_time, load_json, load_presets, load_wordlist,
                    project_dir, rms_db, save_json)

EXIT_NEED_MODEL = 3


def check_platform():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        die("받아쓰기는 애플 실리콘 맥(M1 이상)에서만 돌아가요. "
            f"지금 컴퓨터: {sys.platform}/{platform.machine()}")


def model_cached(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def build_prompt(cfg: dict) -> str:
    vocab = load_wordlist("vocab.txt")
    prompt = cfg["transcribe"]["prompt_example"]
    if vocab:
        prompt += " " + ", ".join(vocab) + "."
    return prompt[:400]   # Whisper 프롬프트는 짧아야 함(약 224토큰)


def chunk_bounds(audio: np.ndarray, sr: int, cfg: dict) -> list[tuple[int, int]]:
    """chunk_minutes 마다, 앞뒤 chunk_search_sec 안에서 가장 조용한 곳으로 나눈다."""
    t = cfg["transcribe"]
    n = len(audio)
    step = int(t["chunk_minutes"] * 60 * sr)
    if n <= step * 1.2:
        return [(0, n)]
    hop = int(sr * cfg["analysis"]["hop_sec"])
    env = rms_db(audio, hop)
    search = int(t["chunk_search_sec"] / cfg["analysis"]["hop_sec"])
    cuts = [0]
    target = step
    while target < n - step * 0.2:
        h = target // hop
        lo, hi = max(0, h - search), min(len(env), h + search)
        best = lo + int(np.argmin(env[lo:hi])) if hi > lo else h
        cut = best * hop
        if cut <= cuts[-1]:
            cut = target
        cuts.append(cut)
        target = cut + step
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


def run_whisper(audio: np.ndarray, cfg: dict, prompt: str) -> dict:
    import mlx_whisper
    t = cfg["transcribe"]
    kw = dict(path_or_hf_repo=t["model"], language=t["language"], word_timestamps=True,
              initial_prompt=prompt, condition_on_previous_text=t["condition_on_previous_text"],
              verbose=None)
    sig = inspect.signature(mlx_whisper.transcribe).parameters
    if "carry_initial_prompt" in sig:          # 프롬프트를 매 구간마다 유지(지원 버전만)
        kw["carry_initial_prompt"] = True
    if "hallucination_silence_threshold" in sig and t.get("hallucination_silence_threshold"):
        kw["hallucination_silence_threshold"] = t["hallucination_silence_threshold"]
    return mlx_whisper.transcribe(audio, **kw)


def simplify(result: dict, offset: float) -> list[dict]:
    segs = []
    for s in result.get("segments", []):
        words = [{"w": w["word"].strip(), "s": round(w["start"] + offset, 3),
                  "e": round(w["end"] + offset, 3), "p": round(float(w.get("probability", 0)), 3)}
                 for w in s.get("words", []) if w.get("word", "").strip()]
        segs.append({"start": round(s["start"] + offset, 3), "end": round(s["end"] + offset, 3),
                     "text": s.get("text", "").strip(), "avg_logprob": s.get("avg_logprob"),
                     "no_speech_prob": s.get("no_speech_prob"), "words": words})
    return segs


def pick_audio(clip: dict, cfg: dict) -> tuple[np.ndarray, str]:
    sr = cfg["analysis"]["sample_rate"]
    src = Path(clip["path"])
    ch = clip["channel"]
    if ch["mode"] == "one_side":
        pcm = ffmpeg_pcm(src, sr, 2)
        return (pcm[:, 0] if ch["use_channel"] == "left" else pcm[:, 1]).copy(), ch["use_channel"]
    # 모노·분리 녹음 모두 섞은 소리로 한 번 받아쓴다.
    # (분리 녹음을 채널별로 따로 받아쓰면 마이크끼리 새어 들어간 소리 때문에 같은 말이 두 번 적힘.
    #  화자는 analyze.py가 좌/우 소리 크기로 단어마다 정함.)
    return ffmpeg_pcm(src, sr, 1)[:, 0].copy(), "mix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--only", help="이 영상 하나만")
    args = ap.parse_args()
    check_platform()
    pdir = project_dir(args.title)
    probe = load_json(pdir / "probe.json") or die("probe.py를 먼저 실행해 주세요.")
    cfg = load_presets(probe["format"])
    t = cfg["transcribe"]
    if not model_cached(t["model"]) and not args.allow_download:
        print(f"NEED_MODEL_DOWNLOAD: 받아쓰기 모델({t['model']}, {t['model_size_note']})이 아직 없어요. "
              "동의해 주시면 --allow-download 로 받아요.")
        sys.exit(EXIT_NEED_MODEL)
    prompt = build_prompt(cfg)
    sr = cfg["analysis"]["sample_rate"]
    clips = [c for c in probe["clips"] if not args.only or c["id"] == args.only]
    total = sum(c["duration"] for c in clips if c["audio"]) or 1
    done_sec = 0.0
    t0 = time.time()
    for i, clip in enumerate(clips, 1):
        out = pdir / "transcripts" / f"{clip['id']}.json"
        prev = load_json(out)
        if prev and prev.get("signature") == clip["signature"] and prev.get("model") == t["model"]:
            print(f"[{i}/{len(clips)}] {clip['id']}: 이미 받아쓴 파일이 있어 건너뜀", flush=True)
            done_sec += clip["duration"]
            continue
        if not clip["audio"]:
            save_json(out, {"clip": clip["id"], "path": clip["path"], "signature": clip["signature"],
                            "model": t["model"], "duration": clip["duration"], "channel_used": None,
                            "segments": [], "note": "오디오 없음"})
            continue
        audio, used = pick_audio(clip, cfg)
        bounds = chunk_bounds(audio, sr, cfg)
        part_dir = pdir / "transcripts" / ".partial" / clip["id"]
        segs = []
        for j, (a, b) in enumerate(bounds):
            pf = part_dir / f"chunk_{j:03d}_{a}_{b}.json"
            part = load_json(pf)
            if part is None or part.get("signature") != clip["signature"]:
                res = run_whisper(audio[a:b], cfg, prompt)
                part = {"signature": clip["signature"], "segments": simplify(res, a / sr)}
                save_json(pf, part)
            segs.extend(part["segments"])
            done = done_sec + b / sr
            el = time.time() - t0
            eta = el / max(done, 1) * (total - done)
            msg = (f"[{i}/{len(clips)}] {clip['id']} 조각 {j + 1}/{len(bounds)} 완료 "
                   f"(전체 {done / total:.0%}, 남은 시간 약 {int(eta // 60)}분)")
            print(msg, flush=True)
            save_json(pdir / "progress.json", {"message": msg, "ratio": done / total})
        save_json(out, {"clip": clip["id"], "path": clip["path"], "signature": clip["signature"],
                        "model": t["model"], "duration": clip["duration"], "channel_used": used,
                        "prompt": prompt, "segments": segs})
        for f in part_dir.glob("*.json"):
            f.unlink()
        part_dir.rmdir()
        done_sec += clip["duration"]
        nwords = sum(len(s["words"]) for s in segs)
        print(f"✅ {clip['id']}: {fmt_time(clip['duration'])} 받아쓰기 끝 (단어 {nwords}개)", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
