#!/usr/bin/env python3
"""영상 정보 확인 + 오디오 채널 구성 판단.

사용:
  probe.py --title 제목 --format vlog --init "/경로/A.MP4" "/경로/B.MP4"   # 처음: batch.txt 저장
  probe.py --title 제목 --format vlog                                       # 이후: batch.txt 사용
  --sort-name : 순서를 안 정했을 때 파일 이름순으로 정렬
"""
from __future__ import annotations

import argparse
import json
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (FORMAT_KO, clip_ids, die, ffmpeg_pcm, file_signature, frame_duration,
                    load_json, load_presets, parse_rate, project_dir, read_batch, resolve_path, rms_db,
                    save_json, timecode_to_frames)


def ffprobe(p: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
                       capture_output=True)
    if r.returncode != 0:
        die(f"영상 정보를 읽지 못했어요: {p}\n{r.stderr.decode(errors='ignore')[-300:]}")
    return json.loads(r.stdout)


def rotation_of(v: dict) -> int:
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if "rotate" in (v.get("tags") or {}):
        rot = int(float(v["tags"]["rotate"]))
    return rot % 360


def find_timecode(info: dict) -> str | None:
    tc = (info.get("format", {}).get("tags") or {}).get("timecode")
    if tc:
        return tc
    for s in info.get("streams", []):
        tc = (s.get("tags") or {}).get("timecode")
        if tc:
            return tc
    return None


def analyze_channels(src: Path, stream_idx: int, cfg: dict) -> dict:
    """좌/우가 같은 소리인지, 마이크 2개가 따로 녹음됐는지."""
    a, c = cfg["analysis"], cfg["channels"]
    sr = a["sample_rate"]
    hop = int(round(sr * a["hop_sec"]))
    pcm = ffmpeg_pcm(src, sr, 2, stream_idx)
    L, R = pcm[:, 0], pcm[:, 1]
    eL, eR = rms_db(L, hop), rms_db(R, hop)
    n = min(len(eL), len(eR))
    eL, eR = eL[:n], eR[:n]
    mix = np.maximum(eL, eR)
    if n == 0:
        return {"mode": "mono", "reason": "오디오가 비어 있음"}
    noise = np.percentile(mix, a["noise_percentile"])
    speech = np.percentile(mix, a["speech_percentile"])
    active = mix > noise + (speech - noise) * a["speech_level_ratio"]
    if active.sum() < 10:
        active = np.ones(n, dtype=bool)
    # 샘플 단위 상관관계(소리 나는 구간만)
    idx = np.repeat(active, hop)[: len(L)]
    l, r = L[: len(idx)][idx], R[: len(idx)][idx]
    corr = float(np.corrcoef(l, r)[0, 1]) if l.std() > 1e-6 and r.std() > 1e-6 else 1.0
    p95L, p95R = float(np.percentile(eL, 95)), float(np.percentile(eR, 95))
    diff = float(np.mean(eL[active]) - np.mean(eR[active]))
    d = eL[active] - eR[active]
    fracL = float(np.mean(d > c["dominance_db"]))
    fracR = float(np.mean(d < -c["dominance_db"]))
    stats = {"corr": round(corr, 3), "level_diff_db": round(diff, 1),
             "left_dominant_ratio": round(fracL, 3), "right_dominant_ratio": round(fracR, 3),
             "p95_left_db": round(p95L, 1), "p95_right_db": round(p95R, 1)}
    if abs(p95L - p95R) >= c["one_side_silent_db"]:
        side = "left" if p95L > p95R else "right"
        return {"mode": "one_side", "use_channel": side, "stats": stats,
                "reason": f"한쪽 채널만 소리가 있어요(차이 {abs(p95L - p95R):.0f}dB) → {'왼쪽' if side == 'left' else '오른쪽'}만 사용"}
    if corr >= c["mono_corr"] and abs(diff) <= c["mono_level_diff_db"]:
        return {"mode": "mono", "stats": stats,
                "reason": f"좌우가 거의 같은 소리(상관 {corr:.2f}, 음량차 {diff:+.1f}dB) → 모노 취급"}
    if fracL >= c["split_side_min_ratio"] and fracR >= c["split_side_min_ratio"]:
        return {"mode": "split", "stats": stats,
                "reason": f"좌우가 다른 마이크로 보여요(상관 {corr:.2f}, 왼쪽 우세 {fracL:.0%} / 오른쪽 우세 {fracR:.0%}) → 화자 구분에 사용"}
    return {"mode": "mono", "stats": stats,
            "reason": f"좌우가 조금 다르지만 화자별로 나뉘진 않아요(상관 {corr:.2f}, 음량차 {diff:+.1f}dB) → 모노 취급"}


def probe_clip(p: Path, cid: str, cfg: dict) -> dict:
    info = ffprobe(p)
    streams = info.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    warn = []
    if not vids:
        die(f"영상 트랙이 없어요: {p.name}")
    v = vids[0]
    fps = parse_rate(v.get("r_frame_rate") or v.get("avg_frame_rate"))
    avg = v.get("avg_frame_rate")
    if avg and avg != "0/0" and abs(float(parse_rate(avg)) - float(fps)) > 0.05:
        warn.append(f"프레임레이트가 일정하지 않을 수 있어요(기준 {float(fps):.3f}, 평균 {float(parse_rate(avg)):.3f})")
    fd = frame_duration(fps)
    rot = rotation_of(v)
    w, h = int(v["width"]), int(v["height"])
    dw, dh = (h, w) if rot in (90, 270) else (w, h)
    dur = float(info["format"].get("duration") or v.get("duration") or 0)
    tc = find_timecode(info)
    tc_frames, tc_drop = 0, False
    if tc:
        try:
            tc_frames, tc_drop = timecode_to_frames(tc, fps), ";" in tc
        except ValueError:
            warn.append(f"타임코드 형식을 못 읽었어요: {tc}")
    clip = {"id": cid, "path": str(p), "signature": file_signature(p),
            "width": dw, "height": dh, "rotation": rot,
            "fps": f"{fps.numerator}/{fps.denominator}", "frame_duration": [fd.numerator, fd.denominator],
            "duration": dur, "timecode": tc, "tc_start_frames": tc_frames, "tc_drop": tc_drop,
            "audio": None, "channel": {"mode": "none", "reason": "오디오 없음"}, "warnings": warn}
    if auds:
        a0 = auds[0]
        ch = int(a0.get("channels") or 0)
        clip["audio"] = {"streams": len(auds), "channels": ch, "rate": int(a0.get("sample_rate") or 48000)}
        if len(auds) > 1:
            warn.append(f"오디오 트랙이 {len(auds)}개예요. 첫 번째 트랙으로 분석해요.")
        if ch == 1:
            clip["channel"] = {"mode": "mono", "reason": "모노 오디오"}
        elif ch >= 2:
            if ch > 2:
                warn.append(f"오디오 채널이 {ch}개예요. 앞의 2개(좌/우)로 판단해요.")
            clip["channel"] = analyze_channels(p, 0, cfg)
    else:
        warn.append("오디오가 없어요. 이 영상은 통째로 남겨요.")
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--format", required=True)
    ap.add_argument("--init", nargs="+", help="영상 경로들(준 순서 그대로)")
    ap.add_argument("--sort-name", action="store_true")
    ap.add_argument("--out-dir", help="결과 저장 폴더(기본: 첫 영상 폴더)")
    args = ap.parse_args()
    cfg = load_presets(args.format)
    pdir = project_dir(args.title)
    pdir.mkdir(parents=True, exist_ok=True)

    if args.init:
        paths = [resolve_path(x) for x in args.init]
        if args.sort_name:
            paths.sort(key=lambda x: unicodedata.normalize("NFC", x.name))
        missing = [str(x) for x in paths if not x.is_file()]
        if missing:
            die("이 파일을 찾지 못했어요:\n  " + "\n  ".join(missing))
        dup = {str(x) for x in paths if paths.count(x) > 1}
        if dup:
            die("같은 파일이 두 번 들어 있어요:\n  " + "\n  ".join(dup))
        (pdir / "batch.txt").write_text("\n".join(str(x) for x in paths) + "\n", encoding="utf-8")
        proj = {"title": args.title, "format": args.format, "created": datetime.now().isoformat(timespec="seconds"),
                "order": "이름순" if args.sort_name else "지정한 순서",
                "out_dir": args.out_dir or str(paths[0].parent)}
        save_json(pdir / "project.json", proj)
    proj = load_json(pdir / "project.json")
    if not proj:
        die("처음 실행할 때는 --init 으로 영상 경로를 넣어 주세요.")
    if proj["format"] != args.format:
        proj["format"] = args.format
        save_json(pdir / "project.json", proj)

    paths = read_batch(pdir)
    ids = clip_ids(paths)
    clips = [probe_clip(p, cid, cfg) for p, cid in zip(paths, ids)]
    first = clips[0]
    tl = {k: first[k] for k in ("width", "height", "fps", "frame_duration")}
    tl["audio_rate"] = (first["audio"] or {}).get("rate", 48000)
    warnings = []
    for c in clips[1:]:
        if c["fps"] != first["fps"]:
            warnings.append(f"{c['id']}: 프레임레이트가 첫 영상과 달라요({c['fps']} ≠ {first['fps']}). 파이널컷이 자동으로 맞춰요.")
        if (c["width"], c["height"]) != (first["width"], first["height"]):
            warnings.append(f"{c['id']}: 해상도가 첫 영상과 달라요({c['width']}x{c['height']}).")
    save_json(pdir / "probe.json", {"title": args.title, "format": args.format, "timeline": tl,
                                    "clips": clips, "warnings": warnings})

    # 쉬운 말 보고
    print(f"📁 작업: {args.title}  ({FORMAT_KO[args.format]}, 영상 {len(clips)}개, 순서: {proj['order']})")
    print(f"🎬 타임라인 기준(첫 영상): {tl['width']}x{tl['height']}, {float(parse_rate(tl['fps'])):.3f}fps")
    total = 0.0
    for c in clips:
        total += c["duration"]
        m, s = divmod(int(c["duration"]), 60)
        print(f"  • {c['id']}: {m}분 {s}초 | {c['width']}x{c['height']} | 오디오: {c['channel']['reason']}")
        for w in c["warnings"]:
            print(f"      ⚠️ {w}")
    for w in warnings:
        print(f"  ⚠️ {w}")
    print(f"⏱ 전체 길이: {int(total // 60)}분 {int(total % 60)}초")
    print(f"PROJECT_DIR={pdir}")


if __name__ == "__main__":
    main()
