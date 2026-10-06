"""피드백 학습: 기록 쌓기, 되살림 비율·제안, 내 설정/내 사전 반영."""
import json

from conftest import W, make_video, seg


def test_feedback_log_and_suggestion(tmp_path, proj, dtd):
    v = make_video(tmp_path / "s" / "a.mp4", 20, [(0, 19)])
    p = proj("피드백")
    p.probe("vlog", [v])
    cid = p.write_transcript(0, [seg(W(f"문장{i}입니다.", i * 2 + 0.1, i * 2 + 1.5)) for i in range(9)])
    p.run("analyze.py")
    refs = [str(n) for n in range(1, 7)]
    p.write("edits.json", {cid: refs})
    p.write("notes.json", {"reasons": {f"{cid}|{r}": {"type": "retake", "reason": "테스트"} for r in refs}})
    p.run("build.py", "--dtd", str(dtd))
    builds = (p.home / "feedback" / "builds.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(builds[-1])["cuts"]["retake"] == 6
    # 반복 컷 6개 중 3개를 되살림 → 50%
    r = p.run("feedback.py", "add", "--kind", "restore", "--refs", "1,2,3", "--request", "1,2,3번 살려줘")
    assert "반복 3" in r.stdout
    log = json.loads((p.home / "feedback" / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert log["items"][0]["text"].startswith("문장0") and log["format"] == "vlog"
    p.run("feedback.py", "add", "--kind", "tighter", "--request", "더 촘촘하게")
    p.run("feedback.py", "add", "--kind", "tighter", "--request", "조금 더 촘촘하게")
    out = p.run("feedback.py", "summary").stdout
    assert "반복: 자름 6 / 되살림 3 (50%)" in out
    assert "retake.similarity" in out          # 반복 기준 조정 제안
    assert "'더 촘촘하게'가 반복됨" in out


def test_my_presets_and_my_dictionaries_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FCP_AUTOCUT_HOME", str(tmp_path))
    (tmp_path / "my_presets.yaml").write_text("vlog:\n  silence:\n    pad_after_sec: 0.05\n", encoding="utf-8")
    (tmp_path / "my_vocab.txt").write_text("치앙마이\n", encoding="utf-8")
    (tmp_path / "my_fillers.txt").write_text("[빼기]\n에\n[확실]\n아오\n", encoding="utf-8")
    from common import load_presets, load_sectioned, load_wordlist
    assert load_presets("vlog")["silence"]["pad_after_sec"] == 0.05
    assert load_presets("narration")["silence"]["pad_after_sec"] == 0.12   # 다른 포맷은 그대로
    assert "치앙마이" in load_wordlist("vocab.txt") and "커밍쏜" in load_wordlist("vocab.txt")
    f = load_sectioned("fillers.txt")
    assert "에" not in f["확실"] and "아오" in f["확실"] and "어" in f["확실"]
