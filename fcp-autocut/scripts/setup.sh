#!/bin/zsh
# fcp-autocut 준비 상태 확인 · 설치
#   setup.sh --check     : 확인만 (아무것도 설치하지 않음)
#   setup.sh --install   : 전용 가상환경 + 파이썬 패키지 설치 (사용자 동의 후에만 실행)
#   setup.sh --youtube   : 유튜브 분석용 yt-dlp 설치/업데이트 (동의 후)
#   setup.sh --dev       : 테스트 도구(pytest) 설치
# 받아쓰기 모델(약 1.6GB)은 여기서 받지 않음 → transcribe.py --allow-download (동의 후)
set -u
MODE="${1:---check}"
HOME_DIR="${FCP_AUTOCUT_HOME:-$HOME/.local/share/fcp-autocut}"
VENV="${HOME_DIR}/venv"
PY="${VENV}/bin/python"
MODEL="mlx-community/whisper-large-v3-turbo"
ok=1

say() { print -r -- "$1"; }

# 1) 애플 실리콘 맥인지
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  say "❌ 이 도구는 애플 실리콘 맥(M1 이상)에서만 돌아가요. 지금 컴퓨터: $(uname -s) $(uname -m)"
  exit 1
fi
say "✅ 애플 실리콘 맥 확인 ($(uname -m))"

# 2) ffmpeg / ffprobe
if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; then
  say "✅ ffmpeg 있음"
else
  say "❌ ffmpeg가 없어요. 터미널에서 다음을 실행해 주세요:  brew install ffmpeg"
  ok=0
fi

# 3) xmllint (맥 기본 포함)
command -v xmllint >/dev/null && say "✅ xmllint 있음" || { say "❌ xmllint가 없어요(맥 기본 도구). Xcode 명령줄 도구 설치: xcode-select --install"; ok=0; }

# 4) 파이널컷 DTD
DTD_DIR="/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources"
latest=$(ls "${DTD_DIR}" 2>/dev/null | grep -E '^FCPXMLv1_[0-9]+\.dtd$' | sed -E 's/FCPXMLv1_([0-9]+)\.dtd/\1/' | sort -n | tail -1)
if [[ -n "${latest}" ]]; then
  say "✅ 파이널컷 FCPXML 규격 1.${latest} 사용"
else
  say "❌ 파이널컷 프로(/Applications/Final Cut Pro.app)에서 FCPXML 규격 파일을 찾지 못했어요."
  ok=0
fi

# 5) 가상환경 + 패키지
install_venv() {
  mkdir -p "${HOME_DIR}"
  if [[ ! -x "${PY}" ]]; then
    if command -v uv >/dev/null; then
      uv venv --python 3.11 "${VENV}" || return 1
    elif command -v python3.11 >/dev/null; then
      python3.11 -m venv "${VENV}" || return 1
    else
      say "❌ 파이썬 3.11이 없어요. 둘 중 하나를 설치해 주세요:  brew install uv   또는   brew install python@3.11"
      return 1
    fi
  fi
  if command -v uv >/dev/null; then
    uv pip install --python "${PY}" "$@"
  else
    "${PY}" -m pip install -q --upgrade pip && "${PY}" -m pip install -q "$@"
  fi
}

case "${MODE}" in
  --install) install_venv mlx-whisper numpy PyYAML || ok=0 ;;
  --youtube) install_venv -U yt-dlp || ok=0 ;;
  --dev)     install_venv pytest || ok=0 ;;
esac

if [[ -x "${PY}" ]]; then
  ver=$("${PY}" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  say "✅ 전용 가상환경: ${VENV} (파이썬 ${ver})"
  missing=$("${PY}" - <<'PYEOF'
import importlib.util as u
print(" ".join(m for m in ("mlx_whisper", "numpy", "yaml") if u.find_spec(m) is None))
PYEOF
)
  if [[ -n "${missing}" ]]; then say "❌ 파이썬 패키지 없음: ${missing}  → setup.sh --install"; ok=0; else say "✅ 파이썬 패키지 준비됨"; fi
  "${PY}" -c 'import yt_dlp' 2>/dev/null && say "✅ 유튜브 분석 도구(yt-dlp) 있음" || say "ℹ️ 유튜브 분석 도구(yt-dlp) 없음 — 필요할 때 setup.sh --youtube"
  if "${PY}" -c "from huggingface_hub import snapshot_download as s; s('${MODEL}', local_files_only=True)" 2>/dev/null; then
    say "✅ 받아쓰기 모델 있음"
  else
    say "ℹ️ 받아쓰기 모델 없음(약 1.6GB) — 첫 받아쓰기 때 동의를 받고 내려받아요"
  fi
else
  say "❌ 전용 가상환경이 없어요 → setup.sh --install"
  ok=0
fi

[[ ${ok} == 1 ]] && say "READY" || { say "NOT_READY"; exit 2; }
