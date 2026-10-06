"""재검토에서 고친 것들: 처리된 후보 인식, 짧은 반복 보호, 인터뷰 멈춤, 마커 상한, 캐시, 집중 보기."""
import json
import xml.etree.ElementTree as ET

from conftest import W, make_video, seg


def test_whole_sentence_cut_handles_partial_candidate(tmp_path, proj, dtd):
    # 후보는 "1:1-1"('어')인데 문장 1을 통째로 잘랐으면 '확인' 마커가 생기면 안 됨
    v = make_video(tmp_path / "s" / "a.mp4", 8, [(0, 3), (4, 7)])
    p = proj("처리확인")
    p.probe("narration", [v])
    cid = p.write_transcript(0, [seg(W("안녕하세요", 0.2, 1.0), W("어", 1.2, 1.4), W("반갑습니다.", 1.6, 2.8)),
                                 seg(W("시작합니다.", 4.2, 6.8))])
    p.run("analyze.py")
    assert any(c.get("refs") == ["1:1-1"] for c in p.json("candidates.json")["candidates"])
    p.write("edits.json", {cid: ["1"]})
    p.run("build.py", "--dtd", str(dtd))
    vals = [m.get("value") for m in ET.parse(v.parent / "처리확인_가편집.fcpxml").getroot().iter("marker")]
    assert not any("확인:" in x for x in vals), vals


def test_short_adjacent_repeat_is_marker_not_cut(tmp_path, proj):
    v = make_video(tmp_path / "s" / "b.mp4", 6, [(0, 5)])
    p = proj()
    p.probe("vlog", [v])
    p.write_transcript(0, [seg(W("진짜", 0.2, 0.6), W("맛있어요.", 0.7, 1.4)),
                           seg(W("진짜", 2.2, 2.6), W("맛있어요.", 2.7, 3.4))])
    p.run("analyze.py")
    r = [c for c in p.json("candidates.json")["candidates"] if c["type"] == "retake"]
    assert r and all(c["action"] == "marker" for c in r)


def test_interview_think_pause_only_on_speaker_change(tmp_path, proj):
    # 왼쪽(진행자) 0~2초, 4~6초 / 오른쪽(출연자) 8~10초. 같은 화자 사이 멈춤은 촘촘히, 화자 바뀔 땐 생각 멈춤 유지
    v = make_video(tmp_path / "s" / "c.mp4", 11, [(0, 2), (4, 6)], [(8, 10)])
    p = proj()
    p.probe("interview", [v])
    p.write("roles.json", {"L": "진행자", "R": "출연자"})
    p.write_transcript(0, [seg(W("첫", 0.1, 0.5), W("질문입니다.", 0.6, 1.9)),
                           seg(W("이어서", 4.1, 4.6), W("묻습니다?", 4.7, 5.9)),
                           seg(W("대답할게요.", 8.1, 9.9))])
    p.run("analyze.py")
    sil = sorted((c["start"], c["end"]) for c in p.json("candidates.json")["candidates"]
                 if c["type"] == "silence" and c["action"] == "cut" and 1.5 < c["start"] < 7)
    assert len(sil) == 2, sil
    same, change = sil
    gap_same_kept = 4.0 - same[1]
    gap_change_kept = 8.0 - change[1]
    assert gap_change_kept > gap_same_kept + 0.5, (same, change)


def test_marker_cap_and_overflow_in_report(tmp_path, proj, dtd):
    v = make_video(tmp_path / "s" / "d.mp4", 30, [(0, 29)])
    p = proj("마커상한")
    p.probe("vlog", [v])
    cid = p.write_transcript(0, [seg(W(f"문장{i}번입니다.", i * 2 + 0.1, i * 2 + 1.5)) for i in range(14)])
    p.run("analyze.py")
    cj = p.json("candidates.json")
    for i in range(20):   # 확신도 다른 가짜 마커 후보 20개
        cj["candidates"].append({"clip": cid, "start": i * 1.4 + 0.2, "end": i * 1.4 + 0.5, "type": "disfluency",
                                 "confidence": 0.5 + i * 0.01, "reason": f"테스트{i}", "action": "marker",
                                 "id": f"t{i}", "refs": []})
    p.write("candidates.json", cj)
    p.write("edits.json", {})
    p.run("build.py", "--dtd", str(dtd))
    todo = [m for m in ET.parse(v.parent / "마커상한_가편집.fcpxml").getroot().iter("marker")
            if m.get("completed") == "0"]
    assert len(todo) == 5                                   # 30초 결과 → 상한 최소 5개
    assert all(int(m.get("value").split("테스트")[1]) >= 15 for m in todo)   # 확신도 높은 것부터
    rep = (v.parent / "마커상한_편집리포트.md").read_text(encoding="utf-8")
    assert "## 마커로 꽂지 않은 확인 후보" in rep and "테스트0 (확신도 0.50)" in rep


def test_audio_decoded_once_and_focus_view(tmp_path, proj, dtd):
    v = make_video(tmp_path / "s" / "e.mp4", 8, [(0, 3), (4, 7)])
    p = proj("캐시")
    p.probe("narration", [v])
    caches = list((p.dir / "cache").glob("*.npz"))
    assert len(caches) == 1                                 # probe가 소리 곡선을 미리 저장
    cid = p.write_transcript(0, [seg(W("하나", 0.2, 0.8), W("어", 0.9, 1.1), W("둘.", 1.2, 2.8)),
                                 seg(W("셋입니다.", 4.2, 6.8))])
    p.run("analyze.py")
    p.write("edits.json", {})
    p.run("build.py", "--dtd", str(dtd))
    assert list((p.dir / "cache").glob("*.npz")) == caches  # 분석·생성 단계에서 새로 디코딩하지 않음
    out = p.run("view.py", "--focus").stdout
    assert "[1]" in out and "군말" in out


def test_folder_input_shot_time_order_and_broken_file(tmp_path, proj):
    import subprocess
    d = tmp_path / "T7 Shield" / "발리 5"
    d.mkdir(parents=True)
    for name, t in [("ZZZ0001", "2026-08-08T01:00:00Z"), ("AAA0002", "2026-08-08T03:00:00Z"),
                    ("MMM0003", "2026-08-08T02:00:00Z")]:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30000/1001",
                        "-f", "lavfi", "-i", "sine=f=220:sample_rate=48000", "-t", "2", "-c:v", "libx264",
                        "-preset", "ultrafast", "-c:a", "aac", "-metadata", f"creation_time={t}", str(d / f"{name}.MP4")],
                       check=True)
    (d / "BROKEN.MP4").write_bytes(b"\0" * 4000)     # 복사가 덜 된 파일
    (d / "ZZZ0001.LRF").write_bytes(b"\0" * 10)       # 포켓3 저화질 사본
    p = proj()
    r = p.run("probe.py", "--format", "vlog", "--init", str(d), "--sort-time")
    order = [x.split("/")[-1] for x in (p.dir / "batch.txt").read_text(encoding="utf-8").split()]
    assert order == ["ZZZ0001.MP4", "MMM0003.MP4", "AAA0002.MP4"]
    assert "BROKEN.MP4" in r.stdout and "촬영 시각순" in r.stdout
