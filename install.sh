#!/bin/zsh
# fcp-autocut 스킬을 맥에 설치: ~/.claude/skills/fcp-autocut 으로 복사
# 이미 설치돼 있으면 내가 고친 설정(config 폴더)은 그대로 두고, 새 버전 설정은 *.new 로 옆에 둠.
set -eu
SRC="${0:A:h}/fcp-autocut"
DST="${HOME}/.claude/skills/fcp-autocut"
mkdir -p "${DST}"
for d in scripts tests; do
  rm -rf "${DST}/${d}"
  cp -R "${SRC}/${d}" "${DST}/${d}"
done
cp "${SRC}/SKILL.md" "${DST}/SKILL.md"
mkdir -p "${DST}/config"
for f in "${SRC}"/config/*; do
  name="${f:t}"
  if [[ -f "${DST}/config/${name}" ]] && ! cmp -s "${f}" "${DST}/config/${name}"; then
    cp "${f}" "${DST}/config/${name}.new"
    print "ℹ️ config/${name}: 내가 고친 버전을 유지하고, 새 버전은 ${name}.new 로 저장했어요."
  else
    cp "${f}" "${DST}/config/${name}"
  fi
done
chmod +x "${DST}"/scripts/*.sh "${DST}"/scripts/*.py
print "✅ 설치 완료: ${DST}"
print "   Claude Code에서 \"fcp-autocut 준비 상태 확인해줘\" 라고 말하면 필요한 도구를 점검해요."
