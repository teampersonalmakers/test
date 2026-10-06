#!/bin/zsh
# fcp-autocut 맥 원클릭 설치
# 사용(터미널에 한 줄):
#   zsh -c "$(curl -fsSL https://raw.githubusercontent.com/teampersonalmakers/test/claude/dazzling-archimedes-zlvnq6/mac-setup.sh)"
# 하는 일: 필요한 도구 확인 → 스킬 내려받아 설치 → 전용 파이썬 환경·패키지 설치 → 맥에서 검사 실행
#          → ~/가편집 전용 작업실(Claude Code 프로젝트) 만들기
#          → 받아쓰기 모델(약 1.6GB)은 물어보고 동의할 때만 내려받음
set -u
BRANCH="claude/dazzling-archimedes-zlvnq6"
TARBALL="https://codeload.github.com/teampersonalmakers/test/tar.gz/refs/heads/${BRANCH}"
SK="${HOME}/.claude/skills/fcp-autocut"
PY="${HOME}/.local/share/fcp-autocut/venv/bin/python"
MODEL="mlx-community/whisper-large-v3-turbo"

step() { print -P "\n%B▶ $1%b"; }
fail() { print -P "\n%F{red}❌ $1%f"; exit 1; }

# 0) 애플 실리콘 맥
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || fail "애플 실리콘 맥(M1 이상)에서만 쓸 수 있어요."

# 1) Homebrew · ffmpeg · uv
step "1/5 필요한 도구 확인"
if ! command -v brew >/dev/null; then
  [[ -x /opt/homebrew/bin/brew ]] && eval "$(/opt/homebrew/bin/brew shellenv)" || fail "Homebrew가 없어요. 먼저 Homebrew를 설치해 주세요."
fi
for t in ffmpeg uv; do
  if ! command -v "${t}" >/dev/null; then
    print "  ${t} 설치 중..."
    brew install "${t}" || fail "${t} 설치에 실패했어요."
  fi
done
print "  ✅ ffmpeg, uv 준비됨"

# 2) 스킬 내려받기 → 설치
step "2/5 스킬 내려받아 설치"
TMP="$(mktemp -d)"
curl -fsSL "${TARBALL}" -o "${TMP}/src.tar.gz" || fail "내려받기에 실패했어요. 인터넷 연결을 확인해 주세요."
tar -xzf "${TMP}/src.tar.gz" -C "${TMP}" || fail "압축을 풀지 못했어요."
SRC="$(find "${TMP}" -maxdepth 1 -type d -name 'test-*' | head -1)"
[[ -f "${SRC}/install.sh" ]] || fail "설치 파일을 찾지 못했어요."
zsh "${SRC}/install.sh" || fail "스킬 설치에 실패했어요."
# 명령 하나(fcp)로 모으기
BIN="${HOME}/.local/share/fcp-autocut/bin"
mkdir -p "${BIN}"
cp "${SK}/scripts/fcp" "${BIN}/fcp" && chmod +x "${BIN}/fcp"
# 가편집 전용 작업실: 지시서 + 확인창 없이 실행되도록 허용 목록
WS="${HOME}/가편집"
mkdir -p "${WS}/.claude"
cp "${SRC}/workspace/CLAUDE.md" "${WS}/CLAUDE.md"
cat > "${WS}/.claude/settings.json" <<JSON
{
  "permissions": {
    "allow": [
      "Bash(~/.local/share/fcp-autocut/bin/fcp:*)",
      "Bash(${BIN}/fcp:*)",
      "Read(~/.local/share/fcp-autocut/**)",
      "Edit(~/.local/share/fcp-autocut/**)",
      "Read(~/.claude/skills/fcp-autocut/**)",
      "Edit(~/.claude/skills/fcp-autocut/**)"
    ],
    "additionalDirectories": ["~/.local/share/fcp-autocut", "~/.claude/skills/fcp-autocut"]
  }
}
JSON
print "  ✅ 가편집 전용 작업실: ${WS}"
rm -rf "${TMP}"

# 3) 전용 파이썬 환경 + 패키지(받아쓰기·유튜브 분석·검사 도구)
step "3/5 전용 파이썬 환경과 패키지 설치 (몇 분 걸려요)"
zsh "${SK}/scripts/setup.sh" --install >/dev/null 2>&1
zsh "${SK}/scripts/setup.sh" --youtube >/dev/null 2>&1
zsh "${SK}/scripts/setup.sh" --dev >/dev/null 2>&1
[[ -x "${PY}" ]] || fail "파이썬 환경을 만들지 못했어요. 이 화면을 캡처해서 Claude에게 보여 주세요."
"${PY}" -c "import mlx_whisper, numpy, yaml, yt_dlp, pytest" 2>/dev/null || fail "일부 패키지 설치에 실패했어요. 이 화면을 캡처해서 Claude에게 보여 주세요."
print "  ✅ 패키지 준비됨"

# 4) 맥에서 검사(파이널컷 규격 파일로 FCPXML 검사 포함)
step "4/5 맥에서 검사 실행 (1분 정도)"
RES="$("${PY}" -m pytest "${SK}/tests" -q -p no:cacheprovider 2>&1 | tail -1)"
print "  결과: ${RES}"
if [[ "${RES}" == *failed* || "${RES}" == *error* ]]; then
  print -P "  %F{yellow}⚠️ 일부 검사가 실패했어요. 이 화면을 캡처해서 Claude에게 보여 주세요.%f"
elif [[ "${RES}" == *skipped* ]]; then
  print -P "  %F{yellow}⚠️ 일부 검사를 건너뛰었어요(파이널컷 프로가 /Applications 에 없으면 규격 검사를 못 해요).%f"
else
  print "  ✅ 모든 검사 통과 (파이널컷 규격 검사 포함)"
fi

# 5) 받아쓰기 모델 — 동의할 때만
step "5/5 받아쓰기 모델"
if "${PY}" -c "from huggingface_hub import snapshot_download as s; s('${MODEL}', local_files_only=True)" 2>/dev/null; then
  print "  ✅ 이미 있어요"
else
  print "  받아쓰기 모델을 한 번 내려받아야 해요. 용량은 약 1.6GB예요."
  read "ans?  지금 받을까요? (y = 받기 / n = 나중에 첫 편집 때) : "
  if [[ "${ans}" == [yY]* ]]; then
    "${PY}" -c "from huggingface_hub import snapshot_download as s; s('${MODEL}')" && print "  ✅ 모델 준비됨" \
      || print -P "  %F{yellow}⚠️ 모델을 받지 못했어요. 첫 편집 때 다시 받을 수 있어요.%f"
  else
    print "  나중에 첫 편집 때 다시 물어볼게요."
  fi
fi

print -P "\n%F{green}%B🎉 설치 끝!%b%f"
print "다음: Claude 앱 → Code → 프로젝트 추가에서 홈 폴더의 '가편집' 폴더를 선택하세요."
print "      그다음부터는 그 프로젝트에서 새 대화 → 영상 경로 붙여넣기만 하면 돼요."
open "${HOME}/가편집" 2>/dev/null
