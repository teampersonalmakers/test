# fcp-autocut — 파이널컷 자동 가편집 스킬 (Claude Code)

영상 경로를 주고 "브이로그로 컷편집해줘"라고 말하면, Claude Code가 받아쓰기 → 대본을 읽고 판단 →
무음·더듬은 말·반복 테이크·촬영 중 잡담을 잘라낸 **파이널컷 XML**과 **편집 리포트**를 원본 폴더에 만들어 줍니다.
원본 영상은 절대 건드리지 않아요.

## 설치 (맥, 처음 한 번)

터미널에서:

```zsh
git clone https://github.com/teampersonalmakers/test.git ~/fcp-autocut-src
cd ~/fcp-autocut-src && git checkout claude/dazzling-archimedes-zlvnq6
zsh install.sh
```

그다음 Claude Code에서 **"fcp-autocut 준비 상태 확인해줘"** → 필요한 도구를 점검하고, 설치가 필요하면 물어본 뒤 설치해요.
(받아쓰기 모델 약 1.6GB는 첫 받아쓰기 때 동의를 받고 내려받아요.)

업데이트할 때는 `git pull` 후 `zsh install.sh`를 다시 실행하면 돼요. 직접 고친 설정 파일(고유명사 사전 등)은 그대로 남아요.

## 쓰는 법

- 가편집: `"/Users/나/Movies/방콕/DJI_0001.MP4" "/Users/나/Movies/방콕/DJI_0002.MP4" 브이로그로 컷편집해줘`
- 파이널컷: **파일 > 가져오기 > XML** → 만들어진 `<제목>_가편집.fcpxml` 선택
- 고치기: `"12번 살려줘"`, `"무음 좀 더 촘촘하게"` → 받아쓰기 없이 바로 다시 만들어요
- 학습: `"이 영상 분석해서 내 편집 스타일 배워줘 https://youtu.be/..."`

## 폴더 구성

```
fcp-autocut/
  SKILL.md              Claude가 따르는 작업 설명서
  config/presets.yaml   포맷별 수치(무음 길이, 여유, 확신도 기준 등)
  config/vocab.txt      고유명사 사전
  config/fillers.txt    군말 사전 ([확실] / [애매])
  config/*.txt          받아쓰기 오류 문장, 감탄사, 제작 대화 키워드
  scripts/              setup.sh autocut.py(준비 한 번에) probe.py transcribe.py analyze.py view.py build.py learn.py
  tests/                합성 영상으로 검사(37개)
install.sh              ~/.claude/skills/fcp-autocut 으로 복사
```

## 맥에서 처음 확인할 것

이 스킬은 리눅스 클라우드 환경에서 만들고 검사했어요. 아래는 맥에서만 확인할 수 있어서, 첫 실행 때 같이 확인합니다.

- 받아쓰기(mlx-whisper) 실제 동작과 속도
- 파이널컷 앱에 들어 있는 규격 파일로 검사: `~/.local/share/fcp-autocut/venv/bin/python -m pytest ~/.claude/skills/fcp-autocut/tests -q`
  (pytest는 `zsh ~/.claude/skills/fcp-autocut/scripts/setup.sh --dev`로 설치)
- 포켓3 영상의 타임코드·세로 영상이 파이널컷에서 정확한 위치로 열리는지
