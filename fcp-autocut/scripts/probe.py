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
import re
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (FORMAT_KO, clip_ids, die, envelope, ffmpeg_pcm, file_signature, frame_duration,
                    load_json, load_presets, parse_rate, project_dir, read_batch, resolve_path, rms_db,
                    save_json, timecode_to_frames)


def ffprobe(p: Path, quiet: bool = False) -> dict | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
                       capture_output=True)
    ok = r.returncode == 0
    info = json.loads(r.stdout) if ok and r.stdout else None
    if ok and info and not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        ok = False
    if not ok:
        if quiet:
            return None
        die(f"영상 정보를 읽지 못했어요: {p}\n{r.stderr.decode(errors='ignore')[-300:]}")
    return info


def shot_time(p: Path, info: dict) -> tuple[float, str]:
    """촬영 시각: 영상에 기록된 creation_time → 없으면 파일 수정 시각."""
    from datetime import datetime as _dt
    tags = dict((info.get("format") or {}).get("tags") or {})
    for st in info.get("streams", []):
        for k, v in (st.get("tags") or {}).items():
            tags.setdefault(k, v)
    ct = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
    if ct:
        try:
            return _dt.fromisoformat(ct.replace("Z", "+00:00")).timestamp(), "촬영 시각"
        except ValueError:
            pass
    return p.stat().st_mtime, "파일 수정 시각"


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


def analyze_channels(src: Path, stream_idx: int, cfg: dict, pdir: Path | None = None, cid: str | None = None) -> dict:
    """좌/우가 같은 소리인지, 마이크 2개가 따로 녹음됐는지."""
    a, c = cfg["analysis"], cfg["channels"]
    sr = a["sample_rate"]
    hop = int(round(sr * a["hop_sec"]))
    pcm = ffmpeg_pcm(src, sr, 2, stream_idx)
    if pdir is not None and stream_idx == 0:
        envelope(src, pdir, cid, cfg, stereo=True, pcm=pcm)   # 분석 단계에서 다시 디코딩하지 않게 미리 저장
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


def probe_clip(p: Path, cid: str, cfg: dict, pdir: Path | None = None) -> dict:
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
            clip["channel"] = analyze_channels(p, 0, cfg, pdir, cid)
    else:
        warn.append("오디오가 없어요. 이 영상은 통째로 남겨요.")
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--format", required=True)
    ap.add_argument("--init", nargs="+", help="영상 경로들(준 순서 그대로)")
    ap.add_argument("--sort-name", action="store_true")
    ap.add_argument("--sort-time", action="store_true", help="영상에 기록된 촬영 시각순(폴더째 줄 때 기본)")
    ap.add_argument("--out-dir", help="결과 저장 폴더(기본: 첫 영상 폴더)")
    args = ap.parse_args()
    cfg = load_presets(args.format)
    pdir = project_dir(args.title)
    pdir.mkdir(parents=True, exist_ok=True)

    if args.init:
        paths = []
        for x in args.init:   # 폴더를 주면 그 안의 영상(.MP4/.MOV)을 모두
            px = resolve_path(x)
            if px.is_dir():
                paths += [f for f in px.iterdir() if f.is_file() and f.suffix.lower() in (".mp4", ".mov")
                          and not f.name.startswith("._")]
            else:
                paths.append(px)
        missing = [str(x) for x in paths if not x.is_file()]
        if missing:
            die("이 파일을 찾지 못했어요:\n  " + "\n  ".join(missing))
        seen = set()
        paths = [x for x in paths if not (str(x) in seen or seen.add(str(x)))]   # 같은 파일 중복 제거
        skipped, infos = [], {}
        for x in list(paths):   # 읽을 수 없는(복사 덜 됨·깨진) 파일은 빼고 진행
            info = ffprobe(x, quiet=True)
            if info is None:
                skipped.append(x.name)
                paths.remove(x)
            else:
                infos[str(x)] = info
        if not paths:
            die("읽을 수 있는 영상이 없어요.")
        if args.sort_name:
            paths.sort(key=lambda x: unicodedata.normalize("NFC", x.name))
            order = "파일 이름순"
        elif args.sort_time:
            times = {str(x): shot_time(x, infos[str(x)]) for x in paths}
            paths.sort(key=lambda x: (times[str(x)][0], x.name))
            srcs = {v[1] for v in times.values()}
            order = "촬영 시각순" if srcs == {"촬영 시각"} else "촬영 시각순(일부는 파일 수정 시각 기준)"
        else:
            order = "지정한 순서"
        (pdir / "batch.txt").write_text("\n".join(str(x) for x in paths) + "\n", encoding="utf-8")
        proj = {"title": args.title, "format": args.format, "created": datetime.now().isoformat(timespec="seconds"),
                "order": order, "skipped": skipped,
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
    prev = load_json(pdir / "probe.json")
    if prev and not args.init and [c["path"] for c in prev["clips"]] == [str(x) for x in paths] and \
            all(c["signature"] == file_signature(Path(c["path"])) for c in prev["clips"]):
        clips = prev["clips"]   # 이미 확인한 영상 그대로 → 다시 읽지 않음
    else:
        clips = [probe_clip(p, cid, cfg, pdir) for p, cid in zip(paths, ids)]
    first = clips[0]
    tl = {k: first[k] for k in ("width", "height", "fps", "frame_duration")}
    tl["audio_rate"] = (first["audio"] or {}).get("rate", 48000)
    warnings = []
    stems = {Path(c["path"]).stem for c in clips}
    for c in clips:   # Finder 복사본("이름 1.MP4")일 수 있는 파일
        m = re.fullmatch(r"(.+) \d+", Path(c["path"]).stem)
        if m and m.group(1) in stems:
            warnings.append(f"{c['id']}: '{m.group(1)}'의 복사본일 수 있어요. 같은 영상이면 하나만 남기는 게 좋아요"
                            "(내용이 같으면 '통째로 다시 찍은 영상'으로 자동 판단해요).")
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
    for name in proj.get("skipped", []):
        print(f"  ⚠️ {name}: 영상을 읽을 수 없어 뺐어요(복사가 덜 됐거나 깨진 파일일 수 있어요).")
    print(f"PROJECT_DIR={pdir}")


if __name__ == "__main__":
    main()
