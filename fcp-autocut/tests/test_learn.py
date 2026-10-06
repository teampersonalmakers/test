"""유튜브 학습: 인터넷 없이 계산 부분만 검사(자막 파싱, 통계, 설정 제안)."""
from conftest import W, seg


def test_parse_vtt():
    from learn import parse_vtt
    vtt = """WEBVTT
Kind: captions
Language: ko

00:00:00.000 --> 00:00:02.000
안녕하세요 <c>커밍쏜</c>입니다

00:00:02.000 --> 00:00:04.000
안녕하세요 커밍쏜입니다

00:00:04.000 --> 00:00:06.000
오늘은 퍼메스 이야기
"""
    assert parse_vtt(vtt) == ["안녕하세요 커밍쏜입니다", "오늘은 퍼메스 이야기"]


def test_compute_stats_and_suggest(tmp_path, monkeypatch):
    monkeypatch.setenv("FCP_AUTOCUT_HOME", str(tmp_path))
    from common import load_presets
    from learn import compute_stats, suggest_presets
    cfg = load_presets("vlog")
    segs = [seg(W("안녕하세요.", 0.0, 0.8), W("오늘은", 1.0, 1.4), W("어", 1.45, 1.6), W("방콕입니다.", 1.65, 2.4),
                W("가볼게요.", 2.6, 3.2))]
    s = compute_stats(segs, 60.0, cfg)
    assert abs(s["pause_between_sentences"]["median"] - 0.2) < 1e-6
    assert s["fillers_per_min"]["strong"] == 1.0
    assert "방콕입니다" in s["top_words"]
    sug = suggest_presets([{"format": "vlog", "stats": s}])
    x = sug["vlog"]
    assert abs(x["suggest_pad_after_sec"] + x["suggest_pad_before_sec"] - 0.2) < 1e-3
