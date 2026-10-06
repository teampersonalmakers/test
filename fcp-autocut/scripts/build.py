#!/usr/bin/env python3
"""edits.json + candidates.json → FCPXML(가편집 타임라인) + 편집 리포트.

사용: build.py --title 제목 [--dtd 경로] [--out-dir 폴더]
- 원본 영상은 읽기만 한다(파일을 만들거나 바꾸지 않음).
- 결과: <out_dir>/<제목>_가편집.fcpxml, <제목>_편집리포트.md
"""
from __future__ import annotations

import argparse
import bisect
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from common import (FORMAT_KO, clip_envelope, data_home, die, fmt_time, ftime, load_json, load_presets,
                    project_dir, save_json, to_frames_ceil, to_frames_floor)

FCP_DTD_DIR = Path("/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources")
TYPE_KO = {"silence": "무음", "noise": "잡음", "ambient": "현장음", "hallucination": "받아쓰기오류", "disfluency": "더듬기",
           "filler": "군말", "retake": "반복", "offtalk": "제작대화", "other": "기타 판단"}

# 파이널컷 표준 포맷 이름(확실한 것만). 목록에 없으면 이름 없이 크기·프레임만 적는다(파이널컷이 사용자 지정으로 처리).
_RATE_TAG = {Fraction(1001, 24000): "2398", Fraction(1, 24): "24", Fraction(1, 25): "25",
             Fraction(1001, 30000): "2997", Fraction(1, 30): "30", Fraction(1, 50): "50",
             Fraction(1001, 60000): "5994", Fraction(1, 60): "60"}


def format_name(w, h, fd: Fraction):
    rate = _RATE_TAG.get(fd)
    if rate is None:
        return None
    if (w, h) == (1920, 1080):
        return f"FFVideoFormat1080p{rate}"
    if (w, h) == (1280, 720):
        return f"FFVideoFormat720p{rate}"
    if (w, h) == (3840, 2160):
        return f"FFVideoFormat3840x2160p{rate}"
    return None


# ───────── DTD ─────────

def find_dtd(explicit: str | None) -> tuple[Path, str]:
    if explicit:
        src = Path(explicit)
    else:
        cands = []
        if FCP_DTD_DIR.exists():
            for f in FCP_DTD_DIR.glob("FCPXMLv1_*.dtd"):
                m = re.fullmatch(r"FCPXMLv(\d+)_(\d+)\.dtd", f.name)
                if m:
                    cands.append(((int(m.group(1)), int(m.group(2))), f))
        if not cands:
            die("파이널컷 프로 안의 FCPXML 규격 파일(DTD)을 찾지 못했어요. 파이널컷이 /Applications 에 설치돼 있는지 확인해 주세요.")
        src = max(cands)[1]
    m = re.search(r"FCPXMLv(\d+)_(\d+)\.dtd$", src.name)
    if not m:
        die(f"DTD 파일 이름에서 버전을 못 읽었어요: {src.name}")
    version = f"{m.group(1)}.{m.group(2)}"
    return src, version


def validate(xml_text: str, dtd: Path) -> tuple[bool, str]:
    """xmllint는 경로의 공백·한글을 못 읽을 수 있어 영문 임시 폴더에 둘 다 복사해서 검사."""
    with tempfile.TemporaryDirectory(prefix="fcpautocut_") as td:
        d = Path(td) / "fcpxml.dtd"
        x = Path(td) / "check.fcpxml"
        shutil.copyfile(dtd, d)
        x.write_text(xml_text, encoding="utf-8")
        r = subprocess.run(["xmllint", "--noout", "--dtdvalid", str(d), str(x)], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


# ───────── 컷 계산 ─────────

class ClipCtx:
    def __init__(self, clip, script, cfg, env):
        self.clip = clip
        self.id = clip["id"]
        self.words = script["words"]
        self.sents = {s["n"]: s for s in script["sentences"]}
        self.sent_list = script["sentences"]
        self.dur = clip["duration"]
        self.fd = Fraction(*clip["frame_duration"])
        self.tc = clip["tc_start_frames"]
        self.nframes = to_frames_floor(self.dur, self.fd)
        self.cfg = cfg
        self.env = env
        self.cuts = []   # dict(t0, t1, b0, b1, kind, ref, text, at)
        self.starts = [w["s"] for w in self.words]

    def word_bounds(self, t):
        """t 주변에서 단어를 침범하지 않는 범위(앞 단어 끝, 뒤 단어 시작)."""
        lo = max((w["e"] for w in self.words if w["e"] <= t + 1e-3), default=0.0)
        hi = min((w["s"] for w in self.words if w["s"] >= t - 1e-3), default=self.dur)
        return lo, hi

    def add(self, t0, t1, kind, ref, text="", b0=None, b1=None, at=None):
        t0, t1 = max(0.0, t0), min(self.dur, t1)
        if t1 <= t0:
            return
        b0 = b0 or self.word_bounds(t0)
        b1 = b1 or self.word_bounds(t1)
        self.cuts.append({"t0": t0, "t1": t1, "b0": b0, "b1": b1, "kind": kind, "ref": ref, "text": text,
                          "at": t0 if at is None else at})

    def add_ref(self, ref: str):
        c = self.cfg["cut"]
        sil = self.cfg["silence"]
        pad = c["word_pad_sec"]
        if ref == "*":
            self.add(0, self.dur, "edit", ref, "영상 전체", (0, 0), (self.dur, self.dur))
            return
        m = re.fullmatch(r"@([\d.]+)-([\d.]+)", ref)
        if m:
            self.add(float(m.group(1)), float(m.group(2)), "edit", ref, "")
            return
        m = re.fullmatch(r"(\d+)(?::(\d+)-(\d+))?", ref)
        if not m:
            die(f"edits.json 표기를 이해하지 못했어요: {self.id} '{ref}' (예: \"12\", \"20:0-2\", \"*\", \"@12.3-13.1\")")
        n = int(m.group(1))
        s = self.sents.get(n)
        if s is None:
            die(f"{self.id}에는 {n}번 문장이 없어요. (이 영상의 문장 번호: "
                f"{self.sent_list[0]['n'] if self.sent_list else '-'}~{self.sent_list[-1]['n'] if self.sent_list else '-'})")
        W = self.words
        nwords = s["w1"] - s["w0"] + 1
        a, b = (int(m.group(2)), int(m.group(3))) if m.group(2) else (0, nwords - 1)
        if not (0 <= a <= b < nwords):
            die(f"{self.id} {ref}: {n}번 문장의 단어 번호는 0~{nwords - 1} 이에요.")
        ia, ib = s["w0"] + a, s["w0"] + b
        prev_e = W[ia - 1]["e"] if ia > 0 else 0.0
        next_s = W[ib + 1]["s"] if ib + 1 < len(W) else self.dur
        text = " ".join(w["w"] for w in W[ia:ib + 1])
        whole = not m.group(2) or (a == 0 and b == nwords - 1)
        t0 = max(prev_e, W[ia]["s"] - pad)
        if whole and ia > 0 and W[ia]["s"] - prev_e <= c["next_sentence_join_sec"]:
            # 문장 통째로: 앞 문장 끝(+여유)부터 잘라 사이 무음 조각이 남지 않게
            t0 = min(W[ia]["s"], prev_e + sil["pad_after_sec"])
        if whole:
            # 문장 통째로: 뒤 멈춤까지 같이 지워 앞 문장 → 다음 문장이 자연스럽게 이어지게
            if ib + 1 < len(W) and next_s - W[ib]["e"] <= c["next_sentence_join_sec"]:
                t1 = max(W[ib]["e"], next_s - sil["pad_before_sec"])
            else:
                t1 = min(next_s, W[ib]["e"] + sil["pad_after_sec"])
        elif b == nwords - 1:
            t1 = W[ib]["e"]
        else:
            t1 = max(W[ib]["e"], next_s - pad)
        self.add(t0, t1, "edit", ref, text, (prev_e, W[ia]["s"]), (W[ib]["e"], next_s), at=W[ia]["s"])

    def cut_ratio(self, t0, t1):
        """[t0, t1] 중 자르기로 덮인 비율(경계 보정 전 기준)."""
        if t1 <= t0:
            return 1.0 if any(c["t0"] <= t0 <= c["t1"] for c in self.cuts) else 0.0
        iv = sorted((max(c["t0"], t0), min(c["t1"], t1)) for c in self.cuts if c["t1"] > t0 and c["t0"] < t1)
        tot, cur = 0.0, t0
        for a, b in iv:
            a = max(a, cur)
            if b > a:
                tot += b - a
                cur = b
        return tot / (t1 - t0)

    def quietest(self, t, lo, hi):
        sw = self.cfg["cut"]["snap_window_sec"]
        a, b = max(t - sw, lo), min(t + sw, hi)
        if b <= a or self.env is None:
            return t
        hop = self.env_hop
        i0, i1 = int(a / hop), max(int(b / hop), int(a / hop) + 1)
        seg = self.env[i0:i1]
        if len(seg) == 0:
            return t
        return (i0 + int(np.argmin(seg)) + 0.5) * hop

    def keeps(self):
        """자를 구간 합치기 → 남길 구간(프레임)."""
        if not self.cuts:
            return [(0, self.nframes)]
        cs = sorted(self.cuts, key=lambda c: c["t0"])
        merged = []
        for c in cs:
            if merged and c["t0"] <= merged[-1]["t1"]:
                m = merged[-1]
                if c["t1"] > m["t1"]:
                    m["t1"], m["b1"] = c["t1"], c["b1"]
            else:
                merged.append(dict(c))
        # 경계 보정: 가장 조용한 지점 → 프레임. 남길 쪽을 넉넉히(말 시작·끝 보호)
        frames = []
        for m in merged:
            if m["t0"] <= 0:
                f0 = 0
            else:
                t = self.quietest(m["t0"], *m["b0"])
                f0 = to_frames_ceil(t, self.fd)
            if m["t1"] >= self.dur - 1e-6:
                f1 = self.nframes
            else:
                t = self.quietest(m["t1"], *m["b1"])
                f1 = to_frames_floor(t, self.fd)
            if f1 > f0:
                frames.append([f0, min(f1, self.nframes)])
        ks, cur = [], 0
        for f0, f1 in frames:
            if f0 > cur:
                ks.append([cur, f0])
            cur = max(cur, f1)
        if cur < self.nframes:
            ks.append([cur, self.nframes])
        # 말이 없는 아주 짧은 조각은 버림
        min_keep = self.cfg["cut"]["min_keep_sec"]
        out = []
        for k0, k1 in ks:
            t0, t1 = float(k0 * self.fd), float(k1 * self.fd)
            if t1 - t0 >= min_keep:
                out.append((k0, k1))
                continue
            i = bisect.bisect_right(self.starts, t1)
            has_word = any(min(t1, w["e"]) - max(t0, w["s"]) > (w["e"] - w["s"]) * 0.5
                           for w in self.words[max(0, i - 50):i])
            if not has_word:
                continue
            out.append((k0, k1))
        return out


# ───────── 메인 ─────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--dtd")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    pdir = project_dir(args.title)
    probe = load_json(pdir / "probe.json") or die("probe.py를 먼저 실행해 주세요.")
    script = load_json(pdir / "script.json") or die("analyze.py를 먼저 실행해 주세요.")
    cj = load_json(pdir / "candidates.json")
    edits = load_json(pdir / "edits.json")
    if edits is None:
        die("edits.json이 없어요. 대본을 읽고 판단한 결과를 먼저 저장해 주세요.")
    notes = load_json(pdir / "notes.json", {}) or {}
    proj = load_json(pdir / "project.json", {}) or {}
    fmt = probe["format"]
    cfg = load_presets(fmt)
    dtd, version = find_dtd(args.dtd)
    ids = [c["id"] for c in probe["clips"]]
    unknown = [k for k in edits if k not in ids]
    if unknown:
        die(f"edits.json에 없는 영상 이름이 있어요: {unknown}\n  가능한 이름: {ids}")
    cands = cj["candidates"]
    reasons = notes.get("reasons", {})
    keep_notes = notes.get("keep", {})
    dismiss = set(notes.get("dismiss", []))

    ctxs = []
    for clip in probe["clips"]:
        env, _ = clip_envelope(clip, pdir, cfg)
        ctx = ClipCtx(clip, script["clips"][clip["id"]], cfg, env)
        ctx.env_hop = cfg["analysis"]["hop_sec"]
        uncut = set(notes.get("uncut", []))   # "c12 살려줘": 자동 컷(무음·잡음) 중 되살릴 것
        for c in cands:
            if c["clip"] == ctx.id and c["type"] in ("silence", "noise") and c["action"] == "cut" and c["id"] not in uncut:
                ctx.add(c["start"], c["end"], c["type"], c["id"], "",
                        (c.get("lo", c["start"]), c["start"] + cfg["cut"]["snap_window_sec"]),
                        (c["end"] - cfg["cut"]["snap_window_sec"], c.get("hi", c["end"])))
        for ref in edits.get(ctx.id, []):
            ctx.add_ref(str(ref))
        ctxs.append(ctx)

    # 프레임레이트/포맷
    tl = probe["timeline"]
    tl_fd = Fraction(*tl["frame_duration"])
    formats = {}

    def fmt_id(w, h, fd):
        key = (w, h, fd)
        if key not in formats:
            formats[key] = f"r{len(formats) + 1}"
        return formats[key]

    tl_fmt = fmt_id(tl["width"], tl["height"], tl_fd)
    for ctx in ctxs:
        fmt_id(ctx.clip["width"], ctx.clip["height"], ctx.fd)
    nfmt = len(formats)
    asset_ids = {ctx.id: f"r{nfmt + i + 1}" for i, ctx in enumerate(ctxs)}

    # 남길 구간 + 마커
    timeline = []   # (ctx, k0, k1, tl_offset_frames, tl_dur_frames, markers)
    cursor = 0
    per_clip = {}
    marker_rows = []
    cut_but_kept = []
    placed = set()
    clip_plan = []
    for ctx in ctxs:
        ks = ctx.keeps()
        kept_frames = sum(k1 - k0 for k0, k1 in ks)
        per_clip[ctx.id] = {"orig": ctx.dur, "kept": float(kept_frames * ctx.fd)}
        # 이 영상의 마커 후보
        mk = []
        for c in cands:
            if c["clip"] != ctx.id or c["id"] in dismiss:
                continue
            handled = ctx.cut_ratio(c["start"], c["end"]) >= 0.5   # 이미 (다른 표기로라도) 잘렸음
            if c["action"] == "marker" and not handled:
                mk.append((c["start"], c["end"], f"{TYPE_KO.get(c['type'], c['type'])}: {c['reason']}", False, c["id"],
                           c["confidence"]))
            elif c["action"] == "cut" and c["type"] not in ("silence", "noise", "hallucination", "ambient") and not handled:
                refs = c.get("refs", [])
                kn = next((keep_notes[f"{ctx.id}|{r}"] for r in refs if f"{ctx.id}|{r}" in keep_notes), None) \
                    or keep_notes.get(c["id"])
                if kn is None:   # 자르기 후보인데 판단 없이 남은 것 → 확인 마커(조용히 남기지 않음)
                    mk.append((c["start"], c["end"], f"확인: {TYPE_KO.get(c['type'])} 후보를 남김 — {c['reason']}",
                               False, c["id"], 1.0))
                    cut_but_kept.append(c)
        for m in notes.get("markers", []):
            if m.get("clip") != ctx.id:
                continue
            if "ref" in m and str(m["ref"]).isdigit() and int(m["ref"]) in ctx.sents:
                t = ctx.sents[int(m["ref"])]["start"]
            else:
                t = float(m.get("at", 0))
            mk.append((t, t, m.get("note", "확인"), bool(m.get("done")), "note", 2.0))
        for g in cj["retake_groups"]:
            if g.get("clip") != ctx.id or g.get("keep") is None:
                continue
            choice = next((x for x in notes.get("retake_choices", []) if x.get("group") == g["id"]), None)
            kept_n = int(choice["kept"]) if choice else g["keep"]
            if kept_n in ctx.sents:
                pos = [t["n"] for t in g["takes"]].index(kept_n) + 1 if kept_n in [t["n"] for t in g["takes"]] else "?"
                mk.append((ctx.sents[kept_n]["start"], ctx.sents[kept_n]["end"],
                           f"반복 {len(g['takes'])}개 중 {pos}번째 선택({g['id']})", True, g["id"], 0.0))
        clip_plan.append((ctx, ks, mk))
    # 🔴 할 일 마커가 너무 많으면 확신도 높은 것만(내 메모·확인 마커는 항상 유지)
    result_min = sum(float((k1 - k0) * c.fd) for c, ks, _ in clip_plan for k0, k1 in ks) / 60
    cap = max(5, int(cfg["markers"]["max_todo_per_min"] * result_min + 0.5))
    todo = sorted((m for _, _, mk in clip_plan for m in mk if not m[3] and m[5] < 1.0), key=lambda m: -m[5])
    dropped = {id(m) for m in todo[cap:]}
    overflow = [m for _, _, mk in clip_plan for m in mk if id(m) in dropped]
    for ctx, ks, mk in clip_plan:
        mk = [m for m in mk if id(m) not in dropped]
        for k0, k1 in ks:
            tl_dur = k1 - k0 if ctx.fd == tl_fd else max(1, round(float((k1 - k0) * ctx.fd / tl_fd)))
            markers = []
            used = set()
            for t0, t1, name, done, src, _conf in mk:
                if src != "note" and src in placed:
                    continue
                # 남은 구간과 겹치는 마커만, 겹친 첫 프레임에
                f = max(k0, to_frames_floor(t0, ctx.fd))
                if f >= k1 or to_frames_ceil(t1, ctx.fd) < k0:
                    continue
                if f in used:
                    f = min(f + 1, k1 - 1)
                used.add(f)
                placed.add(src)
                markers.append((f, name, done))
                marker_rows.append((cursor + round(float((f - k0) * ctx.fd / tl_fd)), name, done, ctx.id))
            timeline.append((ctx, k0, k1, cursor, tl_dur, sorted(markers)))
            cursor += tl_dur

    if cursor == 0:
        die("남길 구간이 하나도 없어요. edits.json을 확인해 주세요.")

    # ── FCPXML ──
    build_no = int((load_json(pdir / "build_count.json", {"n": 0}) or {"n": 0})["n"]) + 1
    name = f"{args.title} 가편집" + (f" v{build_no}" if build_no > 1 else "")
    maxlen = cfg["markers"]["max_name_chars"]
    L = ['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE fcpxml>", f'<fcpxml version="{version}">', "  <resources>"]
    for (w, h, fd), fid in formats.items():
        nm = format_name(w, h, fd)
        nm_attr = f' name="{nm}"' if nm else ""
        L.append(f'    <format id="{fid}"{nm_attr} frameDuration="{ftime(1, fd)}" width="{w}" height="{h}"/>')
    use_media_rep = tuple(int(x) for x in version.split(".")) >= (1, 9)
    vs_attr = ' videoSources="1"' if use_media_rep else ""   # 1.8 이하 규격에는 없는 속성
    for ctx in ctxs:
        c = ctx.clip
        url = Path(c["path"]).absolute().as_uri()
        au = c["audio"]
        a_attr = (f' hasAudio="1" audioSources="1" audioChannels="{au["channels"]}" audioRate="{au["rate"]}"'
                  if au else ' hasAudio="0"')
        common = (f'id="{asset_ids[ctx.id]}" name={quoteattr(ctx.id)} start="{ftime(ctx.tc, ctx.fd)}" '
                  f'duration="{ftime(ctx.nframes, ctx.fd)}" hasVideo="1" format="{fmt_id(c["width"], c["height"], ctx.fd)}"'
                  f'{vs_attr}{a_attr}')
        if use_media_rep:
            L.append(f"    <asset {common}>")
            L.append(f'      <media-rep kind="original-media" src={quoteattr(url)}/>')
            L.append("    </asset>")
        else:
            L.append(f"    <asset {common} src={quoteattr(url)}/>")
    L.append("  </resources>")
    tcfmt = "DF" if probe["clips"][0].get("tc_drop") else "NDF"
    rate = tl.get("audio_rate", 48000)
    a_rate = {32000: "32k", 44100: "44.1k", 48000: "48k", 88200: "88.2k", 96000: "96k"}.get(rate, "48k")
    L.append(f"  <event name={quoteattr(name)}>")
    L.append(f"    <project name={quoteattr(name)}>")
    L.append(f'      <sequence format="{tl_fmt}" duration="{ftime(cursor, tl_fd)}" tcStart="0s" tcFormat="{tcfmt}" '
             f'audioLayout="stereo" audioRate="{a_rate}">')
    L.append("        <spine>")
    for ctx, k0, k1, off, tl_dur, markers in timeline:
        dur_str = ftime(tl_dur, tl_fd) if ctx.fd != tl_fd else ftime(k1 - k0, ctx.fd)
        fmt_attr = "" if ctx.fd == tl_fd and (ctx.clip["width"], ctx.clip["height"]) == (tl["width"], tl["height"]) \
            else f' format="{fmt_id(ctx.clip["width"], ctx.clip["height"], ctx.fd)}"'
        open_tag = (f'          <asset-clip ref="{asset_ids[ctx.id]}" offset="{ftime(off, tl_fd)}" name={quoteattr(ctx.id)} '
                    f'start="{ftime(ctx.tc + k0, ctx.fd)}" duration="{dur_str}"{fmt_attr} tcFormat="{tcfmt}" audioRole="dialogue"')
        if not markers:
            L.append(open_tag + "/>")
            continue
        L.append(open_tag + ">")
        for f, nm, done in markers:
            nm = nm if len(nm) <= maxlen else nm[: maxlen - 1] + "…"
            L.append(f'            <marker start="{ftime(ctx.tc + f, ctx.fd)}" duration="{ftime(1, ctx.fd)}" '
                     f'value={quoteattr(nm)} completed="{1 if done else 0}"/>')
        L.append("          </asset-clip>")
    L += ["        </spine>", "      </sequence>", "    </project>", "  </event>", "</fcpxml>", ""]
    xml = "\n".join(L)

    out_dir = Path(args.out_dir or proj.get("out_dir") or Path(probe["clips"][0]["path"]).parent)
    safe_title = re.sub(r'[/\\:]', "_", args.title)
    xml_path = out_dir / f"{safe_title}_가편집.fcpxml"
    rep_path = out_dir / f"{safe_title}_편집리포트.md"
    srcs = {Path(c["path"]).resolve() for c in probe["clips"]}
    for p in (xml_path, rep_path):
        if p.resolve() in srcs:
            die("결과 파일 경로가 원본과 같아요. 중단합니다.")
    (pdir / "last_build.fcpxml").write_text(xml, encoding="utf-8")
    ok, msg = validate(xml, dtd)
    if not ok:
        die(f"FCPXML 규격 검사(DTD {version})를 통과하지 못했어요. 원본 폴더에는 저장하지 않았어요.\n{msg[:1500]}")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(xml, encoding="utf-8")
    except OSError as e:
        die(f"결과를 저장하지 못했어요({out_dir}): {e}\n  --out-dir 로 다른 폴더를 지정해 주세요.")
    save_json(pdir / "build_count.json", {"n": build_no})

    # ── 리포트 ──
    report = make_report(args.title, fmt, probe, ctxs, per_clip, cursor * tl_fd, timeline, marker_rows, cj,
                         edits, notes, cands, cut_but_kept, tl_fd, name, version, overflow)
    report, counts = report
    rep_path.write_text(report, encoding="utf-8")
    # 편집 기록(피드백 학습용): 어떤 유형을 몇 개 잘랐는지 쌓아 둔다
    from datetime import datetime
    import json as _json
    log = data_home() / "feedback" / "builds.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(_json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "title": args.title, "format": fmt,
                             "build": build_no, "orig_sec": round(sum(v["orig"] for v in per_clip.values()), 1),
                             "result_sec": round(float(cursor * tl_fd), 1), "cuts": counts,
                             "markers": len(marker_rows)}, ensure_ascii=False) + "\n")
    save_json(pdir / "last_build.json", {"xml": str(xml_path), "report": str(rep_path), "event": name,
                                         "orig_sec": sum(v["orig"] for v in per_clip.values()),
                                         "result_sec": float(cursor * tl_fd), "markers": len(marker_rows)})
    tot_o = sum(v["orig"] for v in per_clip.values())
    print(f"✅ 저장: {xml_path}")
    print(f"📄 리포트: {rep_path}")
    print(f"⏱ {fmt_time(tot_o)} → {fmt_time(float(cursor * tl_fd))}  (마커 {len(marker_rows)}개, DTD {version} 검사 통과)")
    print(f"🗂 파이널컷 이벤트 이름: {name}")


def tl_pos(timeline, ctx, t, tl_fd):
    """원본 시각 t가 결과 타임라인 어디쯤인지(잘린 곳이면 그 직후 위치)."""
    f = to_frames_floor(t, ctx.fd)
    end_before = 0
    for c, k0, k1, off, tl_dur, _ in timeline:
        if c._index > ctx._index or (c is ctx and k0 > f):
            break
        if c is ctx and k0 <= f < k1:
            return float((off + round(float((f - k0) * ctx.fd / tl_fd))) * tl_fd)
        end_before = off + tl_dur
    return float(end_before * tl_fd)


def make_report(title, fmt, probe, ctxs, per_clip, total_tl, timeline, marker_rows, cj, edits, notes, cands,
                cut_but_kept, tl_fd, event_name, version, overflow=()):
    for i, c in enumerate(ctxs):
        c._index = i
    tot_o = sum(v["orig"] for v in per_clip.values())
    tot_k = float(total_tl)
    R = [f"# {title} 편집 리포트", "",
         f"- 포맷: {FORMAT_KO[fmt]}",
         f"- 원본 {fmt_time(tot_o)} → 결과 **{fmt_time(tot_k)}** ({tot_k / max(tot_o, 1e-9):.0%} 남김, "
         f"{fmt_time(tot_o - tot_k)} 줄임)",
         f"- 파이널컷 이벤트 이름: `{event_name}` (FCPXML {version})",
         f"- 마커 {len(marker_rows)}개: 🔴 할 일(확인 필요) {sum(1 for m in marker_rows if not m[2])}개 / "
         f"🟢 완료(참고용) {sum(1 for m in marker_rows if m[2])}개", ""]
    # 영상별
    R += ["## 영상별", "", "| 영상 | 원본 | 남김 | 비율 | 경고 |", "|---|---|---|---|---|"]
    for ctx in ctxs:
        v = per_clip[ctx.id]
        r = v["kept"] / max(v["orig"], 1e-9)
        w = "⚠️ 0초 남음(전체 삭제)" if v["kept"] <= 0 else "⚠️ 30% 미만 남음 — 확인 필요" if r < 0.3 else ""
        R.append(f"| {ctx.id} | {fmt_time(v['orig'])} | {fmt_time(v['kept'])} | {r:.0%} | {w} |")
    for w in probe.get("warnings", []) + cj.get("warnings", []):
        R.append(f"- ⚠️ {w}")
    # 유형별 컷
    R += ["", "## 유형별 컷", ""]
    reasons = notes.get("reasons", {})

    def ref_type(cid, ref):
        rr = reasons.get(f"{cid}|{ref}")
        if isinstance(rr, dict) and rr.get("type"):
            return rr["type"], rr.get("reason", "")
        if isinstance(rr, str):
            return "other", rr
        for c in cands:
            if c["clip"] == cid and ref in c.get("refs", []):
                return c["type"], c["reason"]
        return "other", ""

    from collections import defaultdict
    by_type = defaultdict(list)
    sil_n = defaultdict(lambda: [0, 0.0])
    noise_rows = []
    for ctx in ctxs:
        for c in ctx.cuts:
            if c["kind"] == "silence":
                sil_n[ctx.id][0] += 1
                sil_n[ctx.id][1] += c["t1"] - c["t0"]
            elif c["kind"] == "noise":
                noise_rows.append((ctx, c))
            else:
                ty, why = ref_type(ctx.id, c["ref"])
                by_type[ty].append((ctx, c, why))
    tot_sil = sum(v[0] for v in sil_n.values())
    R.append(f"- 무음: {tot_sil}곳 (약 {fmt_time(sum(v[1] for v in sil_n.values()))})")
    if noise_rows:
        R.append(f"- 잡음(말 없는 소리): {len(noise_rows)}곳")
    for ty, rows in by_type.items():
        R.append(f"- {TYPE_KO.get(ty, ty)}: {len(rows)}곳")
    # 삭제 문장 목록
    R += ["", "## 삭제한 말 (분류별)", "",
          "> 살리고 싶은 게 있으면 \"[번호] 살려줘\"라고 말해 주세요. 받아쓰기 없이 바로 다시 만들어요.", ""]
    for ty, rows in by_type.items():
        R.append(f"### {TYPE_KO.get(ty, ty)}")
        for ctx, c, why in sorted(rows, key=lambda x: (x[0]._index, x[1]["t0"])):
            pos = tl_pos(timeline, ctx, c["at"], tl_fd)
            R.append(f"- `[{c['ref']}]` {ctx.id} {fmt_time(c['at'])} (결과 {fmt_time(pos)} 부근) — "
                     f"\"{c['text'][:60]}\"" + (f" · {why}" if why else ""))
        R.append("")
    if noise_rows:
        R += ["### 잡음(말 없이 소리만 있던 곳)", "> 말이었는데 잘렸다면 \"c번호 살려줘\"라고 말해 주세요.", ""]
        for ctx, c in sorted(noise_rows, key=lambda x: (x[0]._index, x[1]["t0"])):
            pos = tl_pos(timeline, ctx, c["t0"], tl_fd)
            R.append(f"- `[{c['ref']}]` {ctx.id} {fmt_time(c['t0'])}~{fmt_time(c['t1'])} (결과 {fmt_time(pos)} 부근)")
        R.append("")
    # 반복 테이크
    groups = cj.get("retake_groups", [])
    if groups:
        R += ["## 반복 테이크 선택", ""]
        for g in groups:
            choice = next((x for x in notes.get("retake_choices", []) if x.get("group") == g["id"]), None)
            kept = choice["kept"] if choice else g["keep"]
            why = choice.get("reason") if choice else g["reason"]
            R.append(f"- **{g['id']}** ({g['clip']}): 남김 `[{kept}]` — {why}")
            for t in g["takes"]:
                sc = f" (점수 {t['score']:.2f})" if "score" in t else ""
                R.append(f"  - `[{t['n']}]`{sc} {t['text'][:60]}")
        R.append("")
    # 마커
    if marker_rows:
        R += ["## 마커 목록 (결과 타임라인 시각)", ""]
        for f, nm, done, cid in sorted(marker_rows):
            R.append(f"- {'🟢' if done else '🔴'} {fmt_time(float(f * tl_fd))} [{cid}] {nm}")
        R.append("")
    if overflow:
        R += [f"## 마커로 꽂지 않은 확인 후보 ({len(overflow)}개)", "",
              "> 마커가 너무 많으면 파이널컷에서 보기 힘들어 확신도 높은 것만 꽂았어요. 나머지는 여기서만 확인하세요.", ""]
        for t0, t1, nm, done, src, conf in sorted(overflow, key=lambda m: m[0]):
            R.append(f"- 원본 {fmt_time(t0)} · {nm} (확신도 {conf:.2f})")
        R.append("")
    # 판단 필요
    asks = [c for c in cands if c.get("ask_user")]
    qs = notes.get("questions", [])
    if asks or qs or cut_but_kept:
        R += ["## 판단이 필요한 항목", ""]
        for q in qs:
            R.append(f"- ❓ {q}")
        for c in asks:
            R.append(f"- ❓ {c['clip']} {fmt_time(c['start'])}: {c['reason']}")
        for c in cut_but_kept:
            R.append(f"- 🔍 {c['clip']} {fmt_time(c['start'])}: 자르기 후보였지만 남김(마커 표시) — {c['reason']}")
        R.append("")
    R += ["## 주의사항", "",
          "- 원본 영상 파일을 옮기거나 이름을 바꾸면 파이널컷에서 연결이 끊겨요(빨간 화면). 그대로 두고 가져오세요.",
          "- 같은 파일을 다시 가져올 때 이벤트 이름이 겹치지 않도록 다시 만들 때마다 v2, v3이 붙어요. 이전 이벤트는 지워도 돼요.",
          "- 말끝이 살짝 먹힌 곳은 클립 끝의 오디오 페이드 핸들을 살짝 끌어 짧은 페이드를 주면 자연스러워요.",
          "- 🔴 할 일 마커는 타임라인 인덱스의 '태그' 목록에서 모아 볼 수 있어요.", ""]
    counts = {"silence": tot_sil, "noise": len(noise_rows), **{ty: len(rows) for ty, rows in by_type.items()}}
    return "\n".join(R), counts


if __name__ == "__main__":
    main()
