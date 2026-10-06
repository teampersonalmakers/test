"""가짜 대본으로: 환각 거르기, 반복 테이크 선택, 현장음 보호, 군말 규칙."""
from conftest import W, make_video, seg


def cands(p, typ=None):
    c = p.json("candidates.json")["candidates"]
    return [x for x in c if typ is None or x["type"] == typ]


def sentences(p):
    sc = p.json("script.json")["clips"]
    return [s for v in sc.values() for s in v["sentences"]]


def test_hallucination_filtered_only_when_no_real_voice(tmp_path, proj):
    # 0~3초 실제 말소리, 5~7초 무음. 무음에 '시청해주셔서 감사합니다' 환각.
    # 10~11초 실제 소리 위의 '감사합니다'는 진짜 말이므로 남겨야 함.
    v = make_video(tmp_path / "src" / "h.mp4", 12, [(0, 3), (10, 11)])
    p = proj()
    p.probe("narration", [v])
    p.write_transcript(0, [
        seg(W("오늘", 0.2, 0.8), W("시작합니다.", 0.9, 2.8)),
        seg(W("시청해주셔서", 5.2, 5.9, 0.3), W("감사합니다", 6.0, 6.8, 0.3)),
        seg(W("감사합니다.", 10.1, 10.9, 0.9)),
        seg(W("ご視聴ありがとう", 8.0, 8.5, 0.2)),
    ])
    p.run("analyze.py")
    texts = [s["text"] for s in sentences(p)]
    assert "시청해주셔서 감사합니다" not in texts
    assert "ご視聴ありがとう" not in texts
    assert "감사합니다." in texts
    h = cands(p, "hallucination")
    assert any("시청해주셔서" in x["text"] and x["ignored_text"] for x in h)


def test_interjection_kept_when_ambiguous(tmp_path, proj):
    # '와!'가 실제 소리 위에 있으면 신뢰도가 낮아도 남김
    v = make_video(tmp_path / "src" / "w.mp4", 6, [(0, 2), (3, 3.6)])
    p = proj()
    p.probe("vlog", [v])
    p.write_transcript(0, [seg(W("여기", 0.1, 0.6), W("왔어요.", 0.7, 1.9)), seg(W("와!", 3.0, 3.5, 0.3))])
    p.run("analyze.py")
    assert "와!" in [s["text"] for s in sentences(p)]


def test_retake_second_of_three_is_best(tmp_path, proj):
    # 같은 말 3번: 1번째 끊김, 2번째 온전, 3번째 끊기고 더듬음 → 2번째를 남김
    v = make_video(tmp_path / "src" / "r.mp4", 20, [(0, 4), (5, 9), (10, 14)])
    p = proj()
    p.probe("narration", [v])
    p.write_transcript(0, [
        seg(W("오늘은", 0.2, 0.7), W("방콕에", 0.8, 1.3), W("왔는데", 1.4, 1.9), W("그러니까", 2.6, 3.2)),
        seg(W("오늘은", 5.2, 5.7), W("방콕에", 5.8, 6.3), W("왔습니다.", 6.4, 7.0), W("정말", 7.1, 7.5),
            W("좋네요.", 7.6, 8.4)),
        seg(W("오늘은", 10.2, 10.7, 0.5), W("방콕에", 10.8, 11.3, 0.5), W("어", 11.4, 11.6, 0.4),
            W("왔는", 12.4, 12.9, 0.4), W("음", 13.0, 13.3, 0.4)),
    ])
    p.run("analyze.py")
    g = p.json("candidates.json")["retake_groups"]
    assert len(g) == 1
    ns = [t["n"] for t in g[0]["takes"]]
    assert len(ns) == 3
    assert g[0]["keep"] == ns[1], g[0]
    cut = {r for c in cands(p, "retake") for r in c["refs"]}
    assert str(ns[0]) in cut and str(ns[2]) in cut and str(ns[1]) not in cut


def test_retake_default_keeps_last(tmp_path, proj):
    v = make_video(tmp_path / "src" / "r2.mp4", 10, [(0, 3), (4, 7)])
    p = proj()
    p.probe("narration", [v])
    p.write_transcript(0, [
        seg(W("이번", 0.2, 0.6), W("영상에서는", 0.7, 1.4), W("편집을", 1.5, 2.0), W("알려드릴게요.", 2.1, 2.9)),
        seg(W("이번", 4.2, 4.6), W("영상에서는", 4.7, 5.4), W("편집을", 5.5, 6.0), W("알려드릴게요.", 6.1, 6.9)),
    ])
    p.run("analyze.py")
    g = p.json("candidates.json")["retake_groups"][0]
    assert g["keep"] == g["takes"][-1]["n"]
    c = cands(p, "retake")[0]
    assert c["action"] == "cut" and c["confidence"] >= 0.8   # 바로 이어서 반복 → 확신도 높음


def test_vlog_ambient_scene_protected(tmp_path, proj):
    # 0~2초 말, 3~9초 현장음(말 없음), 10~12초 말. 앞 1초는 무음.
    v = make_video(tmp_path / "src" / "v.mp4", 13, [(1, 2.5), (3, 9), (10, 12)])
    p = proj()
    p.probe("vlog", [v])
    p.write_transcript(0, [seg(W("바다", 1.1, 1.6), W("왔어요.", 1.7, 2.4)), seg(W("좋다.", 10.2, 11.8))])
    p.run("analyze.py")
    amb = cands(p, "ambient")
    assert amb and all(a["action"] == "keep" for a in amb)
    assert any(a["start"] <= 3.1 and a["end"] >= 8.9 for a in amb)
    sil = [c for c in cands(p, "silence") if c["action"] == "cut"]
    # 현장음 장면 안쪽과 영상 앞 무음(말 없는 앞부분)은 자르지 않음
    assert not any(c["start"] < 9 and c["end"] > 3 for c in sil), sil
    assert not any(c["start"] < 1.0 for c in sil), sil


def test_narration_ambient_is_marker_not_kept_silently(tmp_path, proj):
    v = make_video(tmp_path / "src" / "n.mp4", 10, [(0, 2), (3, 7)])
    p = proj()
    p.probe("narration", [v])
    p.write_transcript(0, [seg(W("설명합니다.", 0.2, 1.8))])
    p.run("analyze.py")
    assert all(a["action"] == "marker" for a in cands(p, "ambient"))


def test_fillers_and_stutters(tmp_path, proj):
    v = make_video(tmp_path / "src" / "f.mp4", 8, [(0, 7)])
    p = proj()
    p.probe("narration", [v])
    p.write_transcript(0, [seg(W("어", 0.2, 0.4), W("이제", 0.45, 0.8), W("출발할게요.", 0.85, 1.6)),
                           seg(W("그", 2.0, 2.2), W("그래서", 2.4, 2.9), W("진짜", 3.0, 3.3), W("진짜", 3.35, 3.7),
                               W("좋아요.", 3.8, 4.5))])
    p.run("analyze.py")
    allc = cands(p)
    by_text = lambda t: [c for c in allc if c["type"] in ("filler", "disfluency") and c["start"] <= t <= c["end"]]
    assert any(c["action"] == "cut" for c in by_text(0.3))          # '어' → 자르기
    assert not by_text(0.6)                                          # '이제' 단독 → 무시(마커도 없음)
    assert by_text(2.1)                                              # '그 그래서' → 후보
    assert not any(c["action"] == "cut" for c in by_text(3.1))      # '진짜 진짜' 강조 → 자르지 않음


def test_interview_guest_disfluency_marker_only(tmp_path, proj):
    # 왼쪽=진행자, 오른쪽=인터뷰이. 인터뷰이의 '어'는 마커만, 진행자의 '어'는 자르기.
    v = make_video(tmp_path / "src" / "i.mp4", 8, [(0, 3)], [(4, 7)])
    p = proj()
    p.probe("interview", [v])
    p.write("roles.json", {"L": "진행자", "R": "인터뷰이"})
    p.write_transcript(0, [seg(W("어", 0.2, 0.4), W("질문할게요?", 0.5, 2.8)),
                           seg(W("어", 4.2, 4.4), W("대답할게요.", 4.5, 6.8))])
    p.run("analyze.py")
    f = [c for c in cands(p) if c["type"] in ("filler", "disfluency")]
    host = [c for c in f if c["start"] < 3]
    guest = [c for c in f if c["start"] > 4]
    assert host and host[0]["action"] == "cut"
    assert guest and guest[0]["action"] == "marker"


def test_offtalk_and_cross_clip_reshoot(tmp_path, proj):
    a = make_video(tmp_path / "src" / "A.mp4", 8, [(0, 7)])
    b = make_video(tmp_path / "src" / "B.mp4", 8, [(0, 7)])
    p = proj()
    p.probe("vlog", [a, b])
    same = [seg(W("오늘은", 0.2, 0.7), W("치앙마이", 0.8, 1.4), W("야시장에", 1.5, 2.1), W("왔습니다.", 2.2, 2.9)),
            seg(W("여기", 3.5, 3.9), W("분위기가", 4.0, 4.6), W("정말", 4.7, 5.0), W("좋아요.", 5.1, 5.9))]
    p.write_transcript(0, same + [seg(W("다시", 6.2, 6.5), W("갈게.", 6.6, 6.9))])
    p.write_transcript(1, same)
    p.run("analyze.py")
    allc = cands(p)
    assert any(c["type"] == "offtalk" for c in allc)
    assert any(c.get("refs") == ["*"] and c["clip"] == "A" for c in allc)
