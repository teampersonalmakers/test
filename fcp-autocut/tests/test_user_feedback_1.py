"""실사용 피드백 #1: 말 중심 컷(개 짖는 소리 등 잡음 자르기), 브이로그 중간 반복의 앞 테이크 자르기."""
from conftest import W, make_video, seg


def cands(p, typ=None):
    return [c for c in p.json("candidates.json")["candidates"] if typ is None or c["type"] == typ]


def test_dog_bark_between_sentences_is_cut_in_vlog(tmp_path, proj):
    # 0~2초 말, 2.6~3.6초 개 짖는 소리(말 아님), 4.2~6초 말
    v = make_video(tmp_path / "s" / "dog.mp4", 7, [(0, 2), (2.6, 3.6), (4.2, 6)])
    p = proj()
    p.probe("vlog", [v])
    p.write_transcript(0, [seg(W("여기", 0.1, 0.6), W("왔어요.", 0.7, 1.9)), seg(W("들어가볼게요.", 4.3, 5.9))])
    p.run("analyze.py")
    noise = [c for c in cands(p, "noise") if c["action"] == "cut"]
    assert noise and noise[0]["start"] < 2.6 and noise[0]["end"] > 3.6, cands(p)
    assert not [c for c in cands(p, "ambient") if c["action"] == "keep" and c["start"] < 4]


def test_vlog_mid_video_retake_with_changed_words_is_cut(tmp_path, proj):
    v = make_video(tmp_path / "s" / "r.mp4", 200, [(0, 199)])
    p = proj()
    p.probe("vlog", [v])
    p.write_transcript(0, [
        seg(W("그럼", 100.1, 100.4), W("이제", 100.5, 100.8), W("숙소로", 100.9, 101.4), W("가볼", 101.5, 101.9),
            W("건데요", 102.0, 102.5)),
        seg(W("자", 103.5, 103.7), W("이제", 103.8, 104.1), W("숙소로", 104.2, 104.7), W("한번", 104.8, 105.1),
            W("가볼게요.", 105.2, 105.9)),
    ])
    p.run("analyze.py")
    r = [c for c in cands(p, "retake") if c["action"] == "cut"]
    assert r and r[0]["refs"] == ["1"], cands(p, "retake")


def test_false_start_fragments_grouped_with_full_sentence(tmp_path, proj):
    v = make_video(tmp_path / "s" / "f.mp4", 20, [(0, 19)])
    p = proj()
    p.probe("narration", [v])
    p.write_transcript(0, [
        seg(W("오늘은", 1.0, 1.5)),                                     # 끊긴 조각1
        seg(W("발리", 2.5, 2.9), W("우붓에", 3.0, 3.5)),                  # 끊긴 조각2
        seg(W("오늘은", 5.0, 5.5), W("발리", 5.6, 6.0), W("우붓에", 6.1, 6.6), W("왔습니다.", 6.7, 7.4)),
    ])
    p.run("analyze.py")
    cut = {r for c in cands(p, "retake") if c["action"] == "cut" for r in c["refs"]}
    assert {"1", "2"} <= cut and "3" not in cut, cands(p, "retake")
