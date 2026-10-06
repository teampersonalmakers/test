"""무음 탐지 · 컷 지점 보정 · 채널 판단 · 받아쓰기 조각 나누기 (합성 오디오)."""
from fractions import Fraction

import numpy as np

from conftest import W, make_video, seg


def test_silence_detected_between_tones(tmp_path, proj):
    # 0~2초 소리, 2~4초 무음, 4~6초 소리, 6~8초 무음
    v = make_video(tmp_path / "src" / "a.mp4", 8, [(0, 2), (4, 6)])
    p = proj()
    p.probe("narration", [v])
    p.write_transcript(0, [seg(W("안녕하세요", 0.2, 1.8)), seg(W("반갑습니다.", 4.2, 5.8))])
    p.run("analyze.py")
    sil = [c for c in p.json("candidates.json")["candidates"] if c["type"] == "silence" and c["action"] == "cut"]
    spans = sorted((c["start"], c["end"]) for c in sil)
    # 말 사이 무음(2~4초) → 앞뒤 여유를 두고 자름
    mid = [s for s in spans if 1.9 < s[0] < 2.3]
    assert mid, spans
    assert mid[0][1] < 4.0 and mid[0][1] > 3.7
    # 끝 무음(6~8초) → 나레이션은 앞뒤 무음도 정리
    assert any(s[0] > 6.0 and s[1] >= 7.9 for s in spans), spans


def test_dynamic_range_warning(tmp_path, proj):
    # 말소리와 배경 차이가 작은 영상 → 경고
    v = make_video(tmp_path / "src" / "noisy.mp4", 6, [(0, 3)])
    import subprocess
    noisy = tmp_path / "src" / "noisy2.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), "-f", "lavfi", "-i",
                    "anoisesrc=a=0.8:d=6:r=48000", "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", str(noisy)], check=True)
    p = proj()
    p.probe("vlog", [noisy])
    p.write_transcript(0, [seg(W("시끄럽네요.", 0.5, 2.5))])
    r = p.run("analyze.py")
    assert "차이가" in r.stdout


def test_snap_moves_to_quietest_point():
    from build import ClipCtx
    env = np.full(1000, -20.0, dtype=np.float32)   # 10초, 10ms 단위
    env[505] = -60.0                                 # 5.05초가 가장 조용
    clip = {"id": "x", "duration": 10.0, "frame_duration": [1001, 30000], "tc_start_frames": 0}
    from common import load_presets
    ctx = ClipCtx(clip, {"words": [], "sentences": []}, load_presets("narration"), env)
    ctx.env_hop = 0.01
    t = ctx.quietest(5.0, 0, 10)
    assert abs(t - 5.055) < 0.006
    # 범위(±80ms) 밖의 더 조용한 지점은 무시
    env[400] = -90.0
    assert abs(ctx.quietest(5.0, 0, 10) - 5.055) < 0.006
    # 단어 경계(lo/hi)를 넘지 않음
    t2 = ctx.quietest(5.0, 4.99, 5.03)
    assert 4.99 <= t2 <= 5.03


def test_channel_mono_split_oneside(tmp_path):
    from common import load_presets
    from probe import analyze_channels
    cfg = load_presets("interview")
    same = make_video(tmp_path / "same.mp4", 6, [(0, 2), (3, 5)])
    assert analyze_channels(same, 0, cfg)["mode"] == "mono"
    # 왼쪽 마이크: 0~2초 말, 오른쪽 마이크: 3~5초 말(마이크 2개 분리)
    split = make_video(tmp_path / "split.mp4", 6, [(0, 2)], [(3, 5)])
    r = analyze_channels(split, 0, cfg)
    assert r["mode"] == "split", r
    one = make_video(tmp_path / "one.mp4", 6, [(0, 2), (3, 5)], [], amp_r=0.0)
    r = analyze_channels(one, 0, cfg)
    assert r["mode"] == "one_side" and r["use_channel"] == "left", r


def test_chunk_bounds_pick_quiet_point():
    from common import load_presets
    from transcribe import chunk_bounds
    cfg = load_presets("vlog")
    cfg["transcribe"]["chunk_minutes"] = 0.5      # 30초 단위
    cfg["transcribe"]["chunk_search_sec"] = 5
    sr = 16000
    t = np.arange(int(sr * 70)) / sr
    a = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    a[int(sr * 32):int(sr * 33)] = 0               # 32~33초가 조용
    b = chunk_bounds(a, sr, cfg)
    assert len(b) >= 2
    first_cut = b[0][1] / sr
    assert 32 <= first_cut <= 33, first_cut
    assert b[0][0] == 0 and b[-1][1] == len(a)
    assert all(x[1] == y[0] for x, y in zip(b, b[1:]))


def test_frame_duration_exact():
    from common import frame_duration
    assert frame_duration(Fraction(24000, 1001)) == Fraction(1001, 24000)
    assert frame_duration(Fraction(25)) == Fraction(1, 25)
    assert frame_duration(Fraction(30000, 1001)) == Fraction(1001, 30000)
    assert frame_duration(Fraction(60000, 1001)) == Fraction(1001, 60000)
    # ffprobe가 반올림된 값을 줘도 정확한 분수로
    assert frame_duration(Fraction(2997, 100)) == Fraction(1001, 30000)


def test_timecode_parse():
    from common import timecode_to_frames
    assert timecode_to_frames("01:00:00:00", Fraction(30000, 1001)) == 108000
    assert timecode_to_frames("00:01:00;02", Fraction(30000, 1001)) == 1800   # 드롭프레임
    assert timecode_to_frames("00:00:01:05", Fraction(25)) == 30
