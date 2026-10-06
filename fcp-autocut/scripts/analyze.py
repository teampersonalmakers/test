#!/usr/bin/env python3
"""후보 계산: 무음 / 현장음 / 환각 / 더듬기 / 반복 테이크 / 군말 / 제작 대화.

사용: analyze.py --title 제목
출력(작업 폴더):
  script.json      : 영상별 단어·문장(번호는 전체 영상에 걸쳐 1부터 고유)
  candidates.json  : 후보 목록(공통 형식) + 반복 묶음 + 소리 수준
스크립트는 '후보'만 만든다. 무음 외의 최종 판단은 Claude가 view.py를 읽고 edits.json에 적는다.
"""
from __future__ import annotations

import argparse
import bisect
import re
from pathlib import Path

import numpy as np

from common import (die, envelope, has_hangul, levels, load_json, load_presets, load_sectioned,
                    load_wordlist, norm, project_dir, runs, save_json)


# ───────── 소리 ─────────

class Audio:
    def __init__(self, env: dict, mode: str, use: str | None, cfg: dict):
        self.hop = env["hop_sec"]
        key = {"left": "l", "right": "r"}.get(use or "", "mono")
        self.e = env.get(key, env["mono"])
        self.l, self.r = env.get("l"), env.get("r")
        self.lv = levels(self.e, cfg)
        a = cfg["analysis"]
        self.speech_gate = self.lv["noise_db"] + self.lv["range_db"] * a["speech_level_ratio"]
        self.thr = self.lv["noise_db"] + self.lv["range_db"] * cfg["silence"]["threshold_ratio"]
        self.n = len(self.e)

    def idx(self, t: float) -> int:
        return int(min(max(round(t / self.hop), 0), self.n))

    def ratio_above(self, s: float, e: float, level: float) -> float:
        a, b = self.idx(s), self.idx(e)
        if b <= a:
            return 0.0
        return float(np.mean(self.e[a:b] >= level))

    def side(self, s: float, e: float, dom: float) -> str | None:
        if self.l is None:
            return None
        a, b = self.idx(s), max(self.idx(e), self.idx(s) + 1)
        d = float(np.mean(self.l[a:b]) - np.mean(self.r[a:b])) if b <= len(self.l) else 0.0
        return "L" if d > dom else "R" if d < -dom else None


def cand(clip, start, end, typ, conf, reason, action, **extra):
    c = {"clip": clip, "start": round(float(start), 3), "end": round(float(end), 3), "type": typ,
         "confidence": round(float(min(conf, 0.99)), 2), "reason": reason, "action": action}
    c.update(extra)
    return c


def action_for(conf: float, cfg: dict) -> str | None:
    c = cfg["confidence"]
    return "cut" if conf >= c["cut"] else "marker" if conf >= c["marker"] else None


# ───────── 환각 ─────────

def detect_hallucinations(cid, segs, audio: Audio, cfg, dicts):
    h = cfg["hallucination"]
    out_flags, cands = [], []
    halls = [norm(x) for x in dicts["hall"]]
    inter = {norm(x) for x in dicts["inter"]}
    vocab = {norm(x) for x in dicts["vocab"]}
    firsts = [s["words"][0]["s"] if s["words"] else None for s in segs]
    lasts = [s["words"][-1]["e"] if s["words"] else None for s in segs]
    prev_end, acc = [], -1e9
    for x in lasts:
        prev_end.append(acc)
        acc = max(acc, x) if x is not None else acc
    next_start, acc = [0.0] * len(segs), 1e9
    for i in range(len(segs) - 1, -1, -1):
        next_start[i] = acc
        acc = min(acc, firsts[i]) if firsts[i] is not None else acc
    for si, s in enumerate(segs):
        ws = s["words"]
        if not ws:
            out_flags.append(False)
            continue
        nt = norm(s["text"]) or norm("".join(w["w"] for w in ws))
        st, en = ws[0]["s"], ws[-1]["e"]
        er = audio.ratio_above(st, max(en, st + audio.hop), audio.speech_gate)
        avgp = float(np.mean([w["p"] for w in ws]))
        nws = [norm(w["w"]) for w in ws]
        maxrep, cur = 1, 1
        for a, b in zip(nws, nws[1:]):
            cur = cur + 1 if a == b and a else 1
            maxrep = max(maxrep, cur)
        phrase = next((x for x in halls if x and x in nt and len(nt) <= len(x) + 6), None)
        foreign = (nt and not has_hangul(s["text"]) and len(ws) <= h["max_foreign_words"] and nt not in vocab)
        prev_e, next_s = prev_end[si], next_start[si]
        isolated = len(ws) == 1 and st - prev_e >= h["isolated_gap_sec"] and next_s - en >= h["isolated_gap_sec"]
        weak = er < h["speech_energy_ratio"]
        signs = []
        if phrase:
            signs.append(f"자주 지어내는 문장('{phrase}')")
        if avgp < h["min_avg_prob"]:
            signs.append(f"신뢰도 낮음({avgp:.2f})")
        if maxrep >= h["repeat_min_count"]:
            signs.append(f"같은 말 {maxrep}번 반복")
        if foreign:
            signs.append("한국어가 아닌 짧은 조각")
        if isolated:
            signs.append("앞뒤로 말이 없는 한 단어")
        is_inter = nt in inter or all(x in inter for x in nws if x)
        flag, conf, reason, action = False, 0.0, "", "keep"
        if is_inter:
            if er < h["interjection_energy_ratio"]:
                flag, conf = True, 0.6
                reason = f"감탄사 '{s['text']}'인데 실제 소리가 거의 없어 받아쓰기 오류로 판단"
            elif signs:
                reason = f"감탄사로 보여 남김('{s['text']}')"
        elif maxrep >= h["loop_repeat_count"]:
            flag, conf = True, 0.85
            reason = f"받아쓰기 오류: 같은 말이 {maxrep}번 반복('{s['text'][:20]}')"
        elif weak and signs:
            flag, conf = True, 0.8
            reason = f"받아쓰기 오류로 판단: {', '.join(signs)}, 실제 말소리 약함({er:.0%})"
        elif phrase and avgp < h["min_avg_prob"]:
            action, conf = "marker", 0.6
            reason = f"받아쓰기 확인 필요: '{s['text'][:20]}'(소리는 있음)"
        out_flags.append(flag)
        if flag or reason:
            # 환각은 '글자'를 무시한다는 뜻. 그 구간의 소리는 무음/현장음 규칙으로 따로 판단.
            cands.append(cand(cid, st, en, "hallucination", conf, reason,
                              "keep" if flag else action, text=s["text"], ignored_text=flag))
    return out_flags, cands


# ───────── 문장 ─────────

ENDING_PUNCT = re.compile(r"[\.\?\!。？！]$")


def build_sentences(cid, words, cfg, start_no):
    sc = cfg["sentence"]
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        split = (nxt is None or ENDING_PUNCT.search(w["w"]) or nxt["s"] - w["e"] >= sc["split_pause_sec"]
                 or len(cur) >= sc["max_words"])
        if split:
            sents.append(cur)
            cur = []
    out = []
    for k, idxs in enumerate(sents):
        ws = [words[i] for i in idxs]
        spk = [w.get("spk") for w in ws if w.get("spk")]
        out.append({"n": start_no + k, "clip": cid, "w0": idxs[0], "w1": idxs[-1],
                    "start": ws[0]["s"], "end": ws[-1]["e"],
                    "text": " ".join(w["w"] for w in ws),
                    "speaker": max(set(spk), key=spk.count) if spk else None})
    return out


def is_complete(text: str, cfg) -> bool:
    t = text.strip()
    if ENDING_PUNCT.search(t):
        return True
    nt = norm(t)
    return any(nt.endswith(e) for e in cfg["sentence"]["endings"])


def refs_for(sents, w_from, w_to):
    """단어 범위 → edits.json 표기 목록. 문장을 통째로 덮으면 "N", 일부면 "N:a-b"."""
    out = []
    for s in sents:
        a, b = max(w_from, s["w0"]), min(w_to, s["w1"])
        if a > b:
            continue
        if a == s["w0"] and b == s["w1"]:
            out.append(str(s["n"]))
        else:
            out.append(f"{s['n']}:{a - s['w0']}-{b - s['w0']}")
    return out


# ───────── 무음·현장음 ─────────

def detect_silence(cid, words, sents, audio: Audio, cfg, fmt, duration):
    sil, sc, amb = cfg["silence"], cfg["silence_common"], cfg["ambient"]
    hop = audio.hop
    out = []
    if audio.n == 0:
        return out
    active = audio.e >= audio.thr
    minblip = int(round(cfg["analysis"]["min_blip_sec"] / hop))
    for a, b in runs(active):
        if b - a < minblip:
            active[a:b] = False
    covered = np.zeros(audio.n, dtype=bool)
    core = np.zeros(audio.n, dtype=bool)
    for w in words:
        covered[audio.idx(w["s"]):audio.idx(w["e"])] = True
        pad = (w["e"] - w["s"]) * sc["word_core_ratio"]
        core[audio.idx(w["s"] + pad):max(audio.idx(w["e"] - pad), audio.idx(w["s"] + pad) + 1)] = True
    # 현장음 / 숨소리
    for a, b in runs(active):
        overlap = covered[a:b].mean() if b > a else 0
        if overlap > sc["ambient_word_overlap"]:
            continue
        length = (b - a) * hop
        peak = float(audio.e[a:b].max())
        if sil["cut_breaths"] and length <= sil["breath_max_sec"] and \
                peak < audio.lv["noise_db"] + audio.lv["range_db"] * sil["breath_peak_ratio"]:
            active[a:b] = False   # 숨소리 → 무음으로 취급
            continue
        if length >= amb["min_sec"]:
            act = amb["action"]
            reason = {"keep": f"현장음 장면 {length:.1f}초(말 없음) → 보호",
                      "marker": f"말 없는 소리 {length:.1f}초 → 필요하면 확인"}.get(act, "")
            out.append(cand(cid, a * hop, b * hop, "ambient", sc["ambient_conf"], reason, act))
    quiet = ~active & ~core
    w_starts = np.array([w["s"] for w in words]) if words else np.zeros(0)
    w_ends = np.array([w["e"] for w in words]) if words else np.zeros(0)
    min_sil = sil["min_silence_sec"]
    for a, b in runs(quiet):
        t0, t1 = a * hop, min(b * hop, duration)
        if t1 - t0 < min_sil:
            continue
        before = w_ends[w_ends <= t0 + 1e-3]
        after = w_starts[w_starts >= t1 - 1e-3]
        prev_e = float(before.max()) if len(before) else None
        next_s = float(after.min()) if len(after) else None
        head = prev_e is None and not active[:a].any()
        tail = next_s is None and not active[b:].any()
        if prev_e is None or next_s is None:
            if not sil["trim_head_tail"]:
                continue
            if head and tail:
                # 영상 전체에 말·소리가 없음
                out.append(cand(cid, t0, t1, "silence", sc["cut_conf"], "영상 전체가 무음", "cut",
                                lo=t0, hi=t1))
                continue
            c0 = t0 if head else t0 + sil["pad_after_sec"]
            c1 = t1 if tail else t1 - sil["pad_before_sec"]
            where = "앞부분" if prev_e is None else "뒷부분"
        else:
            if sil["speech_context_sec"] is not None:
                ctx = sil["speech_context_sec"]
                if t0 - prev_e > ctx or next_s - t1 > ctx:
                    continue   # 말하는 구간 밖(장면) → 건드리지 않음
            if sil["max_cut_gap_sec"] is not None and t1 - t0 > sil["max_cut_gap_sec"]:
                out.append(cand(cid, t0, t1, "silence", sc["long_gap_conf"],
                                f"긴 무음 {t1 - t0:.1f}초 — 장면이면 유지, 아니면 삭제", "marker"))
                continue
            keep_think = 0.0
            if sil["keep_think_pause_sec"]:
                s_prev = next((s for s in sents if abs(s["end"] - prev_e) < 1e-3), None)
                if s_prev is not None:   # 문장 사이 멈춤 → 생각하는 시간 일부 유지
                    keep_think = sil["keep_think_pause_sec"]
            c0 = t0 + sil["pad_after_sec"]
            c1 = t1 - sil["pad_before_sec"] - keep_think
            where = "말 사이"
        if c1 - c0 >= cfg["cut"]["min_cut_sec"]:
            out.append(cand(cid, c0, c1, "silence", sc["cut_conf"], f"{where} 무음 {t1 - t0:.1f}초", "cut",
                            lo=round(t0, 3), hi=round(t1, 3)))
    return out


# ───────── 더듬기 ─────────

def detect_disfluency(cid, words, sents, audio: Audio, cfg, fillers, roles):
    d = cfg["disfluency"]
    C = d["conf"]
    nw = [norm(w["w"]) for w in words]
    strong = {norm(x) for x in fillers.get("확실", [])}
    weak = {norm(x) for x in fillers.get("애매", [])}
    emph = {norm(x) for x in d["emphatic_words"]}
    raw = []   # (w_from, w_to | None, t0, t1, conf, reason)
    weak_sig = []   # 애매한 군말: 다른 신호의 확신도만 올리고, 자를 범위는 넓히지 않음

    def add_w(i, j, conf, reason):
        raw.append((i, j, words[i]["s"], words[j]["e"], conf, reason))

    w2s = [None] * len(words)
    for s in sents:
        for k in range(s["w0"], s["w1"] + 1):
            w2s[k] = s
    for i, w in enumerate(nw):
        s_i = w2s[i]
        # (a) 군말
        if w in strong:
            add_w(i, i, C["strong_filler"], f"군말 '{words[i]['w']}'")
        elif w in weak:
            pb = words[i]["s"] - words[i - 1]["e"] if i > 0 else 9
            pa = words[i + 1]["s"] - words[i]["e"] if i + 1 < len(words) else 9
            bonus = C["weak_filler_pause_bonus"] if max(pb, pa) >= d["weak_filler_pause_sec"] else 0
            weak_sig.append((words[i]["s"], words[i]["e"], C["weak_filler"] + bonus, f"군말일 수 있음 '{words[i]['w']}'"))
        if i == 0 or s_i is None or w2s[i - 1] is not s_i:
            continue
        prev = nw[i - 1]
        # (a) 같은 단어 연속
        if w and w == prev:
            conf = C["emphatic_repeat"] if w in emph else C["word_repeat"]
            add_w(i - 1, i - 1, conf, f"같은 단어 반복 '{words[i - 1]['w']} {words[i]['w']}'")
        # (a) 끊긴 말 "그 그래서"
        elif prev and len(prev) <= d["partial_max_chars"] and w.startswith(prev):
            gap = words[i]["s"] - words[i - 1]["e"]
            conf = C["partial_restart"]
            if gap >= d["partial_pause_sec"] or words[i - 1]["p"] < d["partial_low_prob"]:
                conf += C["partial_restart_bonus"]
            add_w(i - 1, i - 1, conf, f"말 끊고 다시 시작 '{words[i - 1]['w']} {words[i]['w']}'")
    # (a) 구절 재시작 "그래서 이제 … 그래서 이제 우리가"
    taken = set()
    for k in range(6, 1, -1):
        for i in range(len(nw) - k):
            if i in taken or not all(nw[i:i + k]) or all(x in emph for x in nw[i:i + k]):
                continue
            for j in range(i + k, min(len(nw) - k + 1, i + d["restart_max_distance_words"] + 1)):
                if nw[j:j + k] == nw[i:i + k] and words[j]["s"] - words[i]["s"] <= d["restart_max_gap_sec"]:
                    add_w(i, j - 1, C["phrase_restart"],
                          f"같은 말로 다시 시작 '{' '.join(x['w'] for x in words[i:i + k])}…'")
                    taken.update(range(i, j))
                    break
    # (b) 소리는 있는데 받아쓴 글자가 없는 구간
    for s in sents:
        for i in range(s["w0"], s["w1"]):
            g0, g1 = words[i]["e"], words[i + 1]["s"]
            if g1 - g0 >= d["gap_min_sec"] and audio.ratio_above(g0, g1, audio.speech_gate) >= d["gap_speech_ratio"]:
                raw.append((None, None, g0, g1, C["energy_gap"],
                            f"받아쓰기에 없는 말소리 {g1 - g0:.1f}초(더듬었을 가능성)"))
    # (c) 저신뢰 단어 몰림
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j]["p"] < d["low_prob"]:
            j += 1
        if j - i >= d["low_prob_min_words"]:
            add_w(i, j - 1, C["low_prob_cluster"], f"알아듣기 어려운 구간(단어 {j - i}개)")
        i = max(j, i + 1)
    # 겹치거나 붙은 신호 합치기(확신도 = 1 - Π(1 - c))
    raw.sort(key=lambda x: (x[2], x[3]))
    merged = []
    for r in raw:
        if merged and r[2] <= merged[-1]["t1"] + d["merge_gap_sec"]:
            m = merged[-1]
            m["t1"] = max(m["t1"], r[3])
            m["keep"] *= (1 - r[4])
            if r[5] not in m["reasons"]:
                m["reasons"].append(r[5])
        else:
            merged.append({"t0": r[2], "t1": r[3], "keep": 1 - r[4], "reasons": [r[5]]})
    for t0, t1, conf, reason in weak_sig:
        g = next((m for m in merged if t0 < m["t1"] and t1 > m["t0"]), None) or \
            next((m for m in merged if t0 <= m["t1"] + d["merge_gap_sec"] and t1 >= m["t0"] - d["merge_gap_sec"]), None)
        if g is not None:
            g["keep"] *= (1 - conf)
            if reason not in g["reasons"]:
                g["reasons"].append(reason)
        else:
            merged.append({"t0": t0, "t1": t1, "keep": 1 - conf, "reasons": [reason]})
    merged.sort(key=lambda m: m["t0"])
    out = []
    starts = [w["s"] for w in words]
    mode = cfg["disfluency"].get("mode", "cut")
    for m in merged:
        conf = 1 - m["keep"]
        act = action_for(conf, cfg)
        if act is None:
            continue
        lo = bisect.bisect_left(starts, m["t0"] - 1e-3)
        hi = bisect.bisect_right(starts, m["t1"] + 1e-3)
        inside = [k for k in range(lo, hi) if words[k]["e"] <= m["t1"] + 1e-3]
        refs = refs_for(sents, inside[0], inside[-1]) if inside else [f"@{m['t0']:.2f}-{m['t1']:.2f}"]
        spk = words[inside[0]].get("spk") if inside else None
        reason = " + ".join(m["reasons"])
        if mode == "by_speaker" and roles.get(spk or "", "") != "host" and act == "cut":
            act = "marker"
            reason += " (인터뷰이 발화라 마커만)"
        out.append(cand(cid, m["t0"], m["t1"], "filler" if all(r.startswith("군말") for r in m["reasons"])
                        else "disfluency", conf, reason, act, refs=refs))
    return out


# ───────── 반복 테이크 ─────────

def ngrams(t: str, n: int) -> set:
    t = norm(t)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))} if t else set()


def take_score(s, words, disfl, cfg):
    r = cfg["retake"]
    W = r["weights"]
    ws = words[s["w0"]:s["w1"] + 1]
    complete = 1.0 if is_complete(s["text"], cfg) else 0.0
    nd = sum(1 for c in disfl if c["start"] >= s["start"] - 1e-3 and c["end"] <= s["end"] + 1e-3)
    avgp = float(np.mean([w["p"] for w in ws])) if ws else 0
    pauses = sum(1 for a, b in zip(ws, ws[1:]) if b["s"] - a["e"] >= r["inner_pause_sec"])
    parts = {"complete": complete, "disfluency": 1 - min(nd / 3, 1), "prob": avgp, "pause": 1 - min(pauses / 2, 1)}
    return round(sum(W[k] * v for k, v in parts.items()), 3), {"끝맺음": bool(complete), "더듬기": nd,
                                                              "신뢰도": round(avgp, 2), "긴멈춤": pauses}


def detect_retakes(cid, words, sents, disfl, cfg, roles, clip_pos, gid_start):
    r = cfg["retake"]
    n = r["ngram"]
    cands_out, groups = [], []
    elig = [s for s in sents if len(norm(s["text"])) >= r["min_chars"]]
    grams = {s["n"]: ngrams(s["text"], n) for s in elig}
    parent = {s["n"]: s["n"] for s in elig}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(elig):
        for b in elig[i + 1:]:
            if b["start"] - a["end"] > r["window_sec"]:
                break
            ga, gb = grams[a["n"]], grams[b["n"]]
            if not ga or not gb:
                continue
            inter = len(ga & gb)
            dice = 2 * inter / (len(ga) + len(gb))
            la, lb = len(norm(a["text"])), len(norm(b["text"]))
            false_start = (la < lb * r["false_start_length_ratio"] and inter / len(ga) >= r["false_start_containment"]
                           and not is_complete(a["text"], cfg))
            if dice >= r["similarity"] or false_start:
                parent[find(b["n"])] = find(a["n"])
    by_root = {}
    for s in elig:
        by_root.setdefault(find(s["n"]), []).append(s)
    gid = gid_start
    order = [s["n"] for s in sents]
    for takes in by_root.values():
        if len(takes) < 2:
            continue
        gid += 1
        takes.sort(key=lambda s: s["start"])
        scored = []
        for s in takes:
            sc, parts = take_score(s, words, disfl, cfg)
            lenratio = len(norm(s["text"])) / max(len(norm(t["text"])) for t in takes)
            scored.append((s, round(sc * (0.7 + 0.3 * lenratio), 3), parts))
        last = scored[-1]
        best = max(scored[:-1], key=lambda x: x[1])
        if best[1] >= last[1] + r["prefer_earlier_margin"]:
            keep, why = best, (f"{scored.index(best) + 1}번째 테이크가 점수가 뚜렷하게 높음"
                               f"({best[1]:.2f} vs 마지막 {last[1]:.2f}: {best[2]})")
        else:
            keep, why = last, "기본 규칙: 마지막 테이크"
        groups.append({"id": f"g{gid}", "clip": cid, "keep": keep[0]["n"], "reason": why,
                       "takes": [{"n": s["n"], "start": s["start"], "end": s["end"], "text": s["text"],
                                  "score": sc, "detail": parts} for s, sc, parts in scored]})
        for idx, (s, sc, parts) in enumerate(scored):
            if s is keep[0]:
                continue
            nxt = scored[idx + 1][0] if idx + 1 < len(scored) else keep[0]
            between = abs(order.index(nxt["n"]) - order.index(s["n"])) - 1
            adjacent = between <= r["adjacent_max_between"]
            conf = r["conf_adjacent"] if adjacent else r["conf_distant"]
            mode = r.get("mode", "all")
            if mode == "edges":
                edge = r["edge_sec"]
                at_edge = (clip_pos["first"] and s["start"] <= edge) or \
                          (clip_pos["last"] and clip_pos["duration"] - s["end"] <= edge)
                if not at_edge:
                    conf -= r["middle_penalty"]
            act = action_for(conf, cfg)
            reason = ("바로 이어서 같은 말 반복" if adjacent else "같은 내용을 다시 말함") + \
                      f" — {scored.index(keep) + 1}/{len(scored)}번째 테이크를 남김"
            if not is_complete(s["text"], cfg):
                reason += " (이 테이크는 말이 끊김)"
            if mode == "host_only" and roles.get(s.get("speaker") or "", "") != "host" and act == "cut":
                act, reason = "marker", reason + " (진행자 질문이 아니라 마커만)"
            if act:
                cands_out.append(cand(cid, s["start"], s["end"], "retake", conf, reason, act,
                                      refs=[str(s["n"])], group=f"g{gid}", keep_ref=str(keep[0]["n"])))
    return cands_out, groups, gid


def detect_cross_clip(clips_data, cfg, gid_start):
    """다음 영상과 겹치는 내용: 통째로 다시 찍음 / 일부 같은 이야기(질문 필요)."""
    r, rs = cfg["retake"], cfg["reshoot"]
    out, groups = [], []
    gid = gid_start
    for A, B in zip(clips_data, clips_data[1:]):
        sa = [s for s in A["sents"] if len(norm(s["text"])) >= r["min_chars"]]
        sb = [s for s in B["sents"] if len(norm(s["text"])) >= r["min_chars"]]
        if not sa or not sb:
            continue
        gb = [(s, ngrams(s["text"], r["ngram"])) for s in sb]
        matches = []
        for s in sa:
            g = ngrams(s["text"], r["ngram"])
            best = max(((2 * len(g & h) / (len(g) + len(h)), t) for t, h in gb if g and h), default=(0, None),
                       key=lambda x: x[0])
            if best[0] >= r["similarity"]:
                matches.append((s, best[1]))
        if not matches:
            continue
        ratio = len(matches) / len(sa)
        if ratio >= rs["clip_match_ratio"]:
            out.append(cand(A["id"], 0, A["duration"], "retake", rs["conf"],
                            f"다음 영상({B['id']})에서 문장의 {ratio:.0%}를 다시 말함 → 통째로 다시 찍은 것 같음",
                            action_for(rs["conf"], cfg) or "marker", refs=["*"], cross_clip=B["id"]))
        else:
            gid += 1
            groups.append({"id": f"g{gid}", "clip": f"{A['id']} ↔ {B['id']}", "keep": None,
                           "reason": "여러 영상에 걸친 같은 이야기 — 어느 쪽을 남길지 사용자에게 물어볼 것",
                           "takes": [{"n": x["n"], "clip": x["clip"], "text": x["text"]} for m in matches for x in m]})
            for s, t in matches:
                out.append(cand(A["id"], s["start"], s["end"], "retake", 0.5,
                                f"{B['id']}의 {t['n']}번과 같은 이야기 — 어느 쪽을 남길지 질문 필요", "marker",
                                refs=[str(s["n"])], group=f"g{gid}", ask_user=True))
    return out, groups


# ───────── 제작 대화 ─────────

def detect_offtalk(cid, sents, cfg, offdict):
    mode = cfg["offtalk"].get("mode", "cut")
    keys = list(offdict.get("세팅", []))
    if mode != "setup_only":
        keys += offdict.get("일반", [])
    keys = [norm(k) for k in keys]
    out = []
    for s in sents:
        nt = norm(s["text"])
        hit = next((k for k in keys if k and k in nt), None)
        if hit:
            conf = cfg["offtalk"]["conf"]
            out.append(cand(cid, s["start"], s["end"], "offtalk", conf,
                            f"촬영 중 제작 대화일 수 있음('{hit}') — 대본 흐름 보고 판단", action_for(conf, cfg) or "marker",
                            refs=[str(s["n"])]))
    return out


# ───────── 메인 ─────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    args = ap.parse_args()
    pdir = project_dir(args.title)
    probe = load_json(pdir / "probe.json") or die("probe.py를 먼저 실행해 주세요.")
    fmt = probe["format"]
    cfg = load_presets(fmt)
    dicts = {"hall": load_wordlist("hallucinations.txt"), "inter": load_wordlist("interjections.txt"),
             "vocab": load_wordlist("vocab.txt")}
    fillers = load_sectioned("fillers.txt")
    offdict = load_sectioned("offtalk.txt")
    roles_raw = load_json(pdir / "roles.json", {}) or {}
    roles = {k: ("host" if "진행" in v or v == "host" else "guest") for k, v in roles_raw.items()}

    script, all_c, all_g, lv_out, warnings = {}, [], [], {}, []
    next_no, gid = 1, 0
    clips_data = []
    nclips = len(probe["clips"])
    for ci, clip in enumerate(probe["clips"]):
        cid = clip["id"]
        tr = load_json(pdir / "transcripts" / f"{cid}.json")
        if tr is None:
            die(f"{cid}의 받아쓰기가 없어요. transcribe.py를 먼저 실행해 주세요.")
        if tr.get("signature") != clip["signature"]:
            warnings.append(f"{cid}: 받아쓰기 이후 원본 파일이 바뀐 것 같아요. 다시 받아쓰기를 권해요.")
        mode = clip["channel"]["mode"]
        if not clip["audio"]:
            script[cid] = {"words": [], "sentences": []}
            clips_data.append({"id": cid, "sents": [], "duration": clip["duration"]})
            continue
        env = envelope(Path(clip["path"]), pdir, cid, cfg, stereo=mode in ("split", "one_side"))
        audio = Audio(env, mode, clip["channel"].get("use_channel"), cfg)
        lv = audio.lv | {"threshold_db": audio.thr}
        if lv["range_db"] < cfg["analysis"]["min_dynamic_range_db"]:
            warnings.append(f"{cid}: 배경 소리와 말소리 차이가 {lv['range_db']:.0f}dB로 작아요. "
                            "무음 판단이 부정확할 수 있어요(시끄러운 현장).")
        lv_out[cid] = {k: round(v, 1) for k, v in lv.items()}
        segs = tr["segments"]
        flags, hcands = detect_hallucinations(cid, segs, audio, cfg, dicts)
        words = []
        for s, f in zip(segs, flags):
            if f:
                continue
            for w in s["words"]:
                w = dict(w)
                if mode == "split":
                    w["spk"] = audio.side(w["s"], w["e"], cfg["speaker"]["word_dominance_db"])
                words.append(w)
        words.sort(key=lambda w: w["s"])
        if mode == "split":   # 판단 못 한 단어는 앞 단어 화자를 따름
            last = None
            for w in words:
                w["spk"] = w["spk"] or last
                last = w["spk"]
        sents = build_sentences(cid, words, cfg, next_no)
        next_no += len(sents)
        script[cid] = {"words": words, "sentences": sents, "channel_mode": mode}
        c_sil = detect_silence(cid, words, sents, audio, cfg, fmt, clip["duration"])
        c_dis = detect_disfluency(cid, words, sents, audio, cfg, fillers, roles)
        pos = {"first": ci == 0, "last": ci == nclips - 1, "duration": clip["duration"]}
        c_ret, groups, gid = detect_retakes(cid, words, sents, c_dis, cfg, roles, pos, gid)
        c_off = detect_offtalk(cid, sents, cfg, offdict)
        all_c += hcands + c_sil + c_dis + c_ret + c_off
        all_g += groups
        clips_data.append({"id": cid, "sents": sents, "duration": clip["duration"]})
    c_x, g_x = detect_cross_clip(clips_data, cfg, gid)
    all_c += c_x
    all_g += g_x
    order = {c["id"]: i for i, c in enumerate(probe["clips"])}
    all_c.sort(key=lambda c: (order[c["clip"]], c["start"], c["end"]))
    for i, c in enumerate(all_c, 1):
        c["id"] = f"c{i}"
    save_json(pdir / "script.json", {"format": fmt, "clips": script})
    save_json(pdir / "candidates.json", {"format": fmt, "levels": lv_out, "warnings": warnings,
                                         "candidates": all_c, "retake_groups": all_g})
    # 요약
    from collections import Counter
    cnt = Counter((c["type"], c["action"]) for c in all_c)
    ko = {"silence": "무음", "ambient": "현장음", "hallucination": "받아쓰기 오류", "disfluency": "더듬기",
          "filler": "군말", "retake": "반복", "offtalk": "제작 대화"}
    act = {"cut": "자르기", "marker": "마커", "keep": "보호/유지"}
    print(f"🔎 분석 끝: 문장 {next_no - 1}개, 후보 {len(all_c)}개, 반복 묶음 {len(all_g)}개")
    for (t, a), n in sorted(cnt.items()):
        print(f"  • {ko.get(t, t)} {act.get(a, a)}: {n}")
    for w in warnings:
        print(f"  ⚠️ {w}")


if __name__ == "__main__":
    main()
