#!/usr/bin/env python3
"""Claude가 읽을 대본 보기 + 초안 만들기.

사용:
  view.py --title 제목                     # 전체
  view.py --title 제목 --from 120 --to 240  # 문장 번호 범위(긴 영상은 나눠 읽기)
  view.py --title 제목 --clip A001          # 영상 하나
  view.py --title 제목 --suggest            # 확신도 높은 후보로 edits.draft.json / notes.draft.json 생성
  view.py --title 제목 --groups             # 반복 묶음만
  view.py --title 제목 --focus              # 후보가 있는 문장 + 앞뒤 1문장만(긴 영상용)
"""
from __future__ import annotations

import argparse

from common import FORMAT_KO, die, fmt_time, load_json, project_dir, save_json

TYPE_KO = {"silence": "무음", "noise": "잡음", "ambient": "현장음", "hallucination": "받아쓰기오류", "disfluency": "더듬기",
           "filler": "군말", "retake": "반복", "offtalk": "제작대화"}
ACT = {"cut": "✂ 자르기", "marker": "📍 마커", "keep": "🛡 유지"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--from", dest="n_from", type=int)
    ap.add_argument("--to", dest="n_to", type=int)
    ap.add_argument("--clip")
    ap.add_argument("--suggest", action="store_true")
    ap.add_argument("--groups", action="store_true")
    ap.add_argument("--focus", action="store_true", help="후보 있는 문장과 앞뒤 1문장만")
    ap.add_argument("--show-silence", action="store_true", help="짧은 무음 컷도 모두 표시")
    args = ap.parse_args()
    pdir = project_dir(args.title)
    script = load_json(pdir / "script.json") or die("analyze.py를 먼저 실행해 주세요.")
    cj = load_json(pdir / "candidates.json")
    probe = load_json(pdir / "probe.json")
    cands, groups = cj["candidates"], cj["retake_groups"]

    if args.suggest:
        return suggest(pdir, cands, groups)
    if args.groups:
        return show_groups(groups)

    print(f"# {args.title} — {FORMAT_KO[probe['format']]}  (문장 번호는 전체 영상에서 고유)")
    print("# 표기: [번호] 시작시각 화자 문장 / 후보는 c번호·유형·확신도·추천·edits 표기·이유")
    for w in cj["warnings"]:
        print(f"# ⚠️ {w}")
    nclips = len(probe["clips"])
    for ci, clip in enumerate(probe["clips"], 1):
        cid = clip["id"]
        if args.clip and cid != args.clip:
            continue
        sc = script["clips"][cid]
        sents = sc["sentences"]
        words = sc["words"]
        if args.n_from or args.n_to:
            sents_v = [s for s in sents if (args.n_from or 0) <= s["n"] <= (args.n_to or 10**9)]
            if not sents_v:
                continue
        else:
            sents_v = sents
        if args.focus:
            hot = set()
            for c in cands:
                if c["clip"] != cid or c["type"] in ("silence", "noise", "ambient"):
                    continue
                for k, s in enumerate(sents_v):
                    if s["start"] < c["end"] + 0.01 and s["end"] > c["start"] - 0.01:
                        hot.update({k - 1, k, k + 1})
            sents_v = [s for k, s in enumerate(sents_v) if k in hot]
        lv = cj["levels"].get(cid, {})
        print(f"\n━━ [영상 {ci}/{nclips}] {cid}  ({fmt_time(clip['duration'])}, 오디오: {clip['channel']['reason']}, "
              f"배경 {lv.get('noise_db', '?')}dB / 말소리 {lv.get('speech_db', '?')}dB) ━━")
        if not sents:
            print("  (받아쓴 말 없음)")
        cc = [c for c in cands if c["clip"] == cid]
        lo_t = sents_v[0]["start"] - 0.01 if sents_v else 0
        hi_t = sents_v[-1]["end"] + 0.01 if sents_v else clip["duration"]
        if not (args.n_from or args.n_to):
            lo_t, hi_t = -1, clip["duration"] + 1
        shown = {s["n"] for s in sents_v}
        events = []
        for s in sents_v:
            events.append((s["start"], 1, "s", s))
        for c in cc:
            if c["end"] < lo_t or c["start"] > hi_t:
                continue
            if args.focus and c["type"] in ("silence", "noise", "ambient", "hallucination"):
                continue
            if c["type"] == "silence" and c["action"] == "cut" and not args.show_silence and c["end"] - c["start"] < 1.0:
                continue
            events.append((c["start"], 0 if c["type"] in ("silence", "noise", "ambient", "hallucination") else 2, "c", c))
        events.sort(key=lambda e: (e[0], e[1]))
        prev_n = None
        for t, _, kind, obj in events:
            if kind == "s":
                if args.focus and prev_n is not None and obj["n"] - prev_n > 1:
                    print(f"   … ({obj['n'] - prev_n - 1}문장 생략)")
                prev_n = obj["n"]
                spk = f" {obj['speaker']}" if obj.get("speaker") else ""
                print(f"[{obj['n']}] {fmt_time(obj['start'])}{spk}  {obj['text']}")
                has_word_cand = any(c["start"] < obj["end"] and c["end"] > obj["start"] and
                                    any(":" in r for r in c.get("refs", [])) for c in cc)
                if has_word_cand:
                    ws = words[obj["w0"]:obj["w1"] + 1]
                    print("      단어: " + " ".join(f"{i}:{w['w']}" for i, w in enumerate(ws)))
            else:
                c = obj
                ty = TYPE_KO.get(c["type"], c["type"])
                if c["type"] in ("silence", "noise", "ambient"):
                    print(f"   {'♪' if c['type'] == 'ambient' else '···'} {c['id']} {fmt_time(c['start'])}–"
                          f"{fmt_time(c['end'])} {ty} {ACT[c['action']]} | {c['reason']}")
                elif c["type"] == "hallucination":
                    print(f"   🚫 {c['id']} {ty} {ACT[c['action']]} | {c['reason']}")
                else:
                    refs = ",".join(c.get("refs", []))
                    extra = f" [묶음 {c['group']}, 남길 후보 {c['keep_ref']}]" if c.get("keep_ref") else ""
                    ask = " ❓사용자에게 물어볼 것" if c.get("ask_user") else ""
                    print(f"      ⚠ {c['id']} {ty} {c['confidence']:.2f} {ACT[c['action']]} \"{refs}\"{extra}{ask} | {c['reason']}")
    if groups and not args.clip and not (args.n_from or args.n_to):
        print()
        show_groups(groups)


def show_groups(groups):
    print("━━ 반복 묶음 ━━")
    for g in groups:
        print(f"🔁 {g['id']} ({g['clip']}) 추천 남김: {g['keep']} — {g['reason']}")
        for t in g["takes"]:
            sc = f" 점수 {t['score']:.2f} {t.get('detail', '')}" if "score" in t else ""
            print(f"    [{t['n']}]{sc}  {t['text'][:60]}")


def suggest(pdir, cands, groups):
    edits, reasons = {}, {}
    for c in cands:
        if c["action"] != "cut" or c["type"] in ("silence", "noise", "hallucination", "ambient"):
            continue
        for r in c.get("refs", []):
            lst = edits.setdefault(c["clip"], [])
            if r not in lst:
                lst.append(r)
            reasons[f"{c['clip']}|{r}"] = {"type": c["type"], "reason": c["reason"], "cand": c["id"]}
    notes = {"reasons": reasons, "keep": {}, "dismiss": [], "uncut": [], "markers": [], "retake_choices": [], "questions": []}
    save_json(pdir / "edits.draft.json", edits)
    save_json(pdir / "notes.draft.json", notes)
    n = sum(len(v) for v in edits.values())
    print(f"📝 초안: 자르기 {n}개 → edits.draft.json / notes.draft.json")
    print("   대본을 읽고 고친 뒤 edits.json / notes.json 으로 저장하세요(초안 그대로 쓰지 말 것).")


if __name__ == "__main__":
    main()
