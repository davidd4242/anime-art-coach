#!/bin/sh
# Run from the evidence directory. Submit only when model access is intended.
set -eu
test -f evidence.json || { echo "Run from the directory containing evidence.json" >&2; exit 2; }
set -- headless '在当前目录读取 evidence.json。先报告尺寸、遍数和 alpha_equal，列出变化最大的三个格子及其坐标。若有查看图像能力，按 crop 编号逐张查看并说明可辨认的笔触；没有图像能力就写“待人工核对”。区分像素变化和审美偏好。只读，不改图、不改代码。'
if [ -n "${DSH_CLI_JS:-}" ]; then
    exec node "$DSH_CLI_JS" "$@"
fi
exec dsh "$@"
