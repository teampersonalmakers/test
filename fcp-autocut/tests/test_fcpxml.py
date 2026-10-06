"""FCPXML: 프레임 정렬(여러 프레임레이트), DTD 검증, 한글·공백 경로, 타임코드, 원본 보호."""
import hashlib
import re
import xml.etree.ElementTree as ET
from fractions import Fraction
from urllib.parse import unquote, urlparse

import pytest

from conftest import W, make_video, seg

T = re.compile(r"^(\d+)(?:/(\d+))?s$")


def tval(s):
    m = T.match(s)
    assert m, s
    return Fraction(int(m.group(1)), int(m.group(2) or 1))


def build_simple(tmp_path, proj, dtd, rate="30000/1001", size="320x180", folder="촬영 원본/3월 방콕",
                 name="브이로그 1일차 #1.mp4", fmt="narration", timecode=None, edits=None):
    v = make_video(tmp_path / folder / name, 10, [(0, 3), (5, 8)], rate=rate, size=size, timecode=timecode)
    p = proj("방콕 브이로그")
    p.probe(fmt, [v])
    cid = p.write_transcript(0, [seg(W("안녕하세요", 0.2, 1.0), W("어", 1.2, 1.4), W("반갑습니다.", 1.6, 2.8)),
                                 seg(W("여기는", 5.2, 5.9), W("방콕입니다.", 6.0, 7.8))])
    p.run("analyze.py")
    p.write("edits.json", edits if edits is not None else {cid: ["1:1-1"]})
    r = p.run("build.py", "--dtd", str(dtd))
    xml = v.parent / "방콕 브이로그_가편집.fcpxml"
    return p, v, xml, r


@pytest.mark.parametrize("rate,fd", [("24000/1001", Fraction(1001, 24000)), ("24", Fraction(1, 24)),
                                     ("25", Fraction(1, 25)), ("30000/1001", Fraction(1001, 30000)),
                                     ("30", Fraction(1, 30)), ("50", Fraction(1, 50)),
                                     ("60000/1001", Fraction(1001, 60000))])
def test_frame_alignment(tmp_path, proj, dtd, rate, fd):
    p, v, xml, _ = build_simple(tmp_path, proj, dtd, rate=rate)
    root = ET.parse(xml).getroot()
    fmt = root.find("resources/format")
    assert tval(fmt.get("frameDuration")) == fd          # 원본 프레임레이트 유지(30으로 바꾸지 않음)
    seq = root.find(".//sequence")
    clips = seq.findall("spine/asset-clip")
    assert len(clips) >= 2
    cursor = Fraction(0)
    for c in clips:
        for attr in ("offset", "start", "duration"):
            x = tval(c.get(attr))
            assert (x / fd).denominator == 1, f"{attr}={c.get(attr)} 가 프레임 경계가 아님({rate})"
        assert tval(c.get("offset")) == cursor              # 빈틈 없이 이어 붙임
        cursor += tval(c.get("duration"))
        for m in c.findall("marker"):
            assert (tval(m.get("start")) / fd).denominator == 1
    assert tval(seq.get("duration")) == cursor


def test_dtd_valid_and_korean_space_path(tmp_path, proj, dtd):
    p, v, xml, r = build_simple(tmp_path, proj, dtd)
    assert "검사 통과" in r.stdout
    root = ET.parse(xml).getroot()
    rep = root.find(".//media-rep")
    src = (rep if rep is not None else root.find("resources/asset")).get("src")   # 1.8 이하는 asset에 직접
    assert src.startswith("file:///")
    assert " " not in src and "%20" in src and "%EC" in src.upper()   # 공백·한글이 인코딩됨
    assert unquote(urlparse(src).path) == str(v.resolve())          # 되돌리면 정확히 원본 경로
    assert (v.parent / "방콕 브이로그_편집리포트.md").exists()


def test_standard_format_name_1080p(tmp_path, proj, dtd):
    _, _, xml, _ = build_simple(tmp_path, proj, dtd, size="1920x1080")
    fmt = ET.parse(xml).getroot().find("resources/format")
    assert fmt.get("name") == "FFVideoFormat1080p2997"


def test_timecode_offset(tmp_path, proj, dtd):
    _, _, xml, _ = build_simple(tmp_path, proj, dtd, timecode="01:00:00:00")
    root = ET.parse(xml).getroot()
    asset = root.find("resources/asset")
    tc = Fraction(108000 * 1001, 30000)
    assert tval(asset.get("start")) == tc
    for c in root.findall(".//asset-clip"):
        assert tval(c.get("start")) >= tc


def test_originals_untouched_and_only_two_outputs(tmp_path, proj, dtd):
    folder = tmp_path / "원본 폴더"
    v = make_video(folder / "A 클립.mp4", 6, [(0, 2), (3, 5)])
    before = hashlib.md5(v.read_bytes()).hexdigest()
    listing = set(x.name for x in folder.iterdir())
    p = proj("원본보호")
    p.probe("vlog", [v])
    cid = p.write_transcript(0, [seg(W("하나", 0.2, 1.8)), seg(W("둘.", 3.2, 4.8))])
    p.run("analyze.py")
    p.write("edits.json", {})
    p.run("build.py", "--dtd", str(dtd))
    assert hashlib.md5(v.read_bytes()).hexdigest() == before
    new = set(x.name for x in folder.iterdir()) - listing
    assert new == {"원본보호_가편집.fcpxml", "원본보호_편집리포트.md"}


def test_whole_clip_removed_and_bad_refs(tmp_path, proj, dtd):
    a = make_video(tmp_path / "s" / "A.mp4", 6, [(0, 2), (3, 5)])
    b = make_video(tmp_path / "s" / "B.mp4", 6, [(0, 2), (3, 5)])
    p = proj("통째로")
    p.probe("narration", [a, b])
    p.write_transcript(0, [seg(W("첫", 0.2, 1.8)), seg(W("번째.", 3.2, 4.8))])
    p.write_transcript(1, [seg(W("두", 0.2, 1.8)), seg(W("번째.", 3.2, 4.8))])
    p.run("analyze.py")
    p.write("edits.json", {"A": ["*"]})
    p.run("build.py", "--dtd", str(dtd))
    root = ET.parse(tmp_path / "s" / "통째로_가편집.fcpxml").getroot()
    names = {c.get("name") for c in root.findall(".//asset-clip")}
    assert names == {"B"}
    rep = (tmp_path / "s" / "통째로_편집리포트.md").read_text(encoding="utf-8")
    assert "0초 남음" in rep
    # 없는 문장 번호 / 없는 영상 이름 → 친절한 오류, 결과 파일 안 바뀜
    p.write("edits.json", {"A": ["99"]})
    r = p.run("build.py", "--dtd", str(dtd), check=False)
    assert r.returncode != 0 and "문장이 없어요" in r.stderr
    p.write("edits.json", {"C": ["1"]})
    r = p.run("build.py", "--dtd", str(dtd), check=False)
    assert r.returncode != 0 and "없는 영상 이름" in r.stderr


def test_rebuild_after_restore_and_markers(tmp_path, proj, dtd):
    # "N번 살려줘": edits.json만 고쳐 다시 build → 결과가 길어지고 이벤트 이름에 v2
    p, v, xml, _ = build_simple(tmp_path, proj, dtd, edits=None)
    cid = p.json("probe.json")["clips"][0]["id"]
    p.write("edits.json", {cid: ["1", "1:1-1"]})
    p.run("build.py", "--dtd", str(dtd))
    d1 = tval(ET.parse(xml).getroot().find(".//sequence").get("duration"))
    p.write("edits.json", {cid: ["1:1-1"]})
    p.write("notes.json", {"markers": [{"clip": cid, "ref": "2", "note": "자막 강조"}]})
    r = p.run("build.py", "--dtd", str(dtd))
    root = ET.parse(xml).getroot()
    d2 = tval(root.find(".//sequence").get("duration"))
    assert d2 > d1
    assert "v3" in root.find("event").get("name")
    vals = [m.get("value") for m in root.findall(".//marker")]
    assert "자막 강조" in vals


def test_undecided_cut_candidate_becomes_todo_marker(tmp_path, proj, dtd):
    # 확신도 높은 자르기 후보를 edits.json에 안 넣고 이유도 없으면 → 조용히 남기지 않고 할 일 마커
    p, v, xml, _ = build_simple(tmp_path, proj, dtd, edits={})
    root = ET.parse(xml).getroot()
    todo = [m for m in root.findall(".//marker") if m.get("completed") == "0"]
    assert any("군말" in m.get("value") for m in todo)
