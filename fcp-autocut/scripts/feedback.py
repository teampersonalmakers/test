#!/usr/bin/env python3
"""피드백을 데이터로 쌓고, 쌓인 피드백으로 설정 조정을 제안.

  feedback.py --title 제목 add --kind restore --refs "12,15:0-2" --request "12번 살려줘, 리액션은 남겨줘"
      kind: restore(살려줘) / cut(이것도 잘라줘) / tighter(더 촘촘하게) / looser(더 느슨하게) / other
      refs가 있으면 그 문장이 어떤 유형으로 잘렸었는지(반복·더듬기 등)와 문장을 자동으로 찾아 함께 기록
  feedback.py summary [--format vlog]
      유형별 "자른 수 대비 되살린 비율", 촘촘/느슨 요청 수, 설정 조정 제안, 최근 요청 원문

저장 위치: ~/.local/share/fcp-autocut/feedback/ (log.jsonl, builds.jsonl). 스킬을 다시 설치해도 지워지지 않음.
제안만 하고 presets.yaml은 바꾸지 않는다(사용자 동의 후 Claude가 수정).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime

from common import FORMAT_KO, data_home, die, load_json, project_dir

TYPE_KO = {"silence": "무음", "disfluency": "더듬기", "filler": "군말", "retake": "반복", "offtalk": "제작대화",
           "other": "기타 판단"}
# 유형별로 되살림이 많을 때 손볼 설정(presets.yaml)
HINT = {
    "retake": "retake.similarity 올리기 / retake.short_chars 올리기 (비슷하지만 다른 말을 반복으로 덜 봄)",
    "disfluency": "disfluency.conf 의 energy_gap·word_repeat 낮추기 (더듬기 판단을 보수적으로)",
    "filler": "fillers.txt 의 [확실] 에서 자주 되살린 말을 [애매]로 옮기기",
    "offtalk": "offtalk.txt 에서 자주 되살린 키워드 빼기",
    "silence": "해당 포맷 silence.pad_before_sec / pad_after_sec 늘리기",
}


def fb_dir():
    d = data_home() / "feedback"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def lookup(title, refs):
    """edits 표기 → 유형·문장 찾기."""
    pdir = project_dir(title)
    script = load_json(pdir / "script.json") or {"clips": {}}
    cj = load_json(pdir / "candidates.json") or {"candidates": []}
    notes = load_json(pdir / "notes.json", {}) or {}
    sents = {s["n"]: (cid, s) for cid, v in script["clips"].items() for s in v["sentences"]}
    out = []
    for ref in refs:
        m = re.match(r"(\d+)", ref)
        if not m or int(m.group(1)) not in sents:
            out.append({"ref": ref, "type": "other", "text": ""})
            continue
        cid, s = sents[int(m.group(1))]
        rr = notes.get("reasons", {}).get(f"{cid}|{ref}")
        ty = rr.get("type") if isinstance(rr, dict) else None
        if not ty:
            ty = next((c["type"] for c in cj["candidates"] if c["clip"] == cid and ref in c.get("refs", [])), "other")
        out.append({"ref": ref, "clip": cid, "type": ty, "text": s["text"][:80]})
    return out


def cmd_add(a):
    proj = load_json(project_dir(a.title) / "project.json") or {}
    refs = [r.strip() for r in (a.refs or "").split(",") if r.strip()]
    rec = {"date": datetime.now().isoformat(timespec="seconds"), "title": a.title, "format": proj.get("format"),
           "kind": a.kind, "request": a.request, "items": lookup(a.title, refs)}
    with (fb_dir() / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tys = Counter(i["type"] for i in rec["items"])
    print(f"📝 피드백 기록: {a.kind} {', '.join(f'{TYPE_KO.get(k, k)} {v}' for k, v in tys.items()) or ''}".rstrip())


def cmd_summary(a):
    logs = [x for x in read_jsonl(fb_dir() / "log.jsonl") if not a.format or x.get("format") == a.format]
    builds = [x for x in read_jsonl(fb_dir() / "builds.jsonl") if not a.format or x.get("format") == a.format]
    if not logs and not builds:
        print("아직 쌓인 피드백이 없어요.")
        return
    # 같은 작업의 마지막 빌드 기준으로 '자른 수'를 센다(다시 만들기 중복 제외)
    last = {}
    for b in builds:
        last[b["title"]] = b
    cut_tot = defaultdict(int)
    for b in last.values():
        for k, v in b["cuts"].items():
            cut_tot[(b["format"], k)] += v
    restored = Counter((x.get("format"), i["type"]) for x in logs if x["kind"] == "restore" for i in x["items"])
    added = Counter((x.get("format"), i["type"]) for x in logs if x["kind"] == "cut" for i in x["items"])
    tl = Counter((x.get("format"), x["kind"]) for x in logs if x["kind"] in ("tighter", "looser"))
    fmts = sorted({f for f, _ in cut_tot} | {f for f, _ in restored} | {f for f, _ in tl} - {None})
    print(f"📊 편집 {len(last)}건, 피드백 {len(logs)}개")
    sugg = []
    for f in fmts:
        print(f"\n[{FORMAT_KO.get(f, f)}]")
        for ty in TYPE_KO:
            c, r = cut_tot.get((f, ty), 0), restored.get((f, ty), 0)
            if c or r:
                rate = r / c if c else 0
                print(f"  • {TYPE_KO[ty]}: 자름 {c} / 되살림 {r} ({rate:.0%})" +
                      (f" / 더 자르라고 함 {added[(f, ty)]}" if added[(f, ty)] else ""))
                if r >= 3 and rate >= 0.2 and ty in HINT:
                    sugg.append(f"[{FORMAT_KO.get(f, f)}] {TYPE_KO[ty]}을 {rate:.0%} 되살림 → {HINT[ty]}")
        t, l = tl[(f, "tighter")], tl[(f, "looser")]
        if t or l:
            print(f"  • 더 촘촘하게 {t}번 / 더 느슨하게 {l}번")
            if t - l >= 2:
                sugg.append(f"[{FORMAT_KO.get(f, f)}] '더 촘촘하게'가 반복됨 → silence.min_silence_sec·pad 값을 기본으로 낮추기")
            if l - t >= 2:
                sugg.append(f"[{FORMAT_KO.get(f, f)}] '더 느슨하게'가 반복됨 → silence.pad 값을 기본으로 늘리기")
    if sugg:
        print("\n🔧 설정 조정 제안(아직 안 바꿈):")
        for s in sugg:
            print(f"  - {s}")
    print("\n🗒 최근 요청 원문(편집취향 노트에 규칙으로 정리할 것):")
    for x in logs[-15:]:
        print(f"  - {x['date'][:10]} [{FORMAT_KO.get(x.get('format'), '?')}] {x['request']}")
    notes = data_home() / "편집취향.md"
    print(f"\n편집취향 노트: {notes} ({'있음' if notes.exists() else '아직 없음'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", help="작업 제목(add에 필요)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("add")
    a.add_argument("--kind", required=True, choices=["restore", "cut", "tighter", "looser", "other"])
    a.add_argument("--refs")
    a.add_argument("--request", required=True)
    s = sp.add_parser("summary")
    s.add_argument("--format")
    args = ap.parse_args()
    if args.cmd == "add" and not args.title:
        die("--title 이 필요해요.")
    {"add": cmd_add, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
