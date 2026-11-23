#!/bin/sh
# Run from the evidence directory. Submit only when model access is intended.
set -eu
test -f evidence.json || { echo "Run from the directory containing evidence.json" >&2; exit 2; }
case "${1:-zh}" in
    zh) prompt='在当前目录读取 evidence.json。先报告尺寸、遍数和 alpha_equal，列出变化最大的三个格子及其坐标。若有查看图像能力，按 crop 编号逐张查看并说明可辨认的笔触；没有图像能力就写“待人工核对”。区分像素变化和审美偏好。只读，不改图、不改代码。' ;;
    en) prompt='Read evidence.json in this directory. Reply in English. Report the size, pass count and alpha_equal, then the three cells with the largest changes and their coordinates. If you can inspect images, review each crop in order and name the surviving strokes. Otherwise mark them as needing human review. Separate pixel changes from artistic preferences. Read only.' ;;
    *) echo "Usage: review.sh [zh|en]" >&2; exit 2 ;;
esac
set -- headless "$prompt"
if [ -n "${DSH_CLI_JS:-}" ]; then
    exec node "$DSH_CLI_JS" "$@"
fi
exec dsh "$@"
