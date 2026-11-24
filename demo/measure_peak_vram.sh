#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "用法：$0 [--int8] [步数上限]（默认使用 FP32，最多 1024 步，在 GPU 0 上运行）"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
infer=./build/qwen3_infer
checkpoint=model/qwen0.6.bin2
if [[ ${1:-} == --int8 ]]; then
    infer=./build-bench/qwen3_infer_int8
    checkpoint=model/qwen0.6.int8.bin
    shift
fi
steps=${1-1024}
if (( $# > 1 )) || ! [[ $steps =~ ^[1-9][0-9]*$ ]]; then
    usage >&2
    exit 2
fi

cd -- "$(dirname -- "$0")/.."
if ! gpu=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader); then
    echo "无法访问 GPU 0。$gpu" >&2
    exit 1
fi
query=(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if ! baseline=$("${query[@]}") || ! [[ $baseline =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
    echo "无法读取 GPU 0 的显存占用：$baseline" >&2
    exit 1
fi

log=$(mktemp)
monitor=
stop_monitor() {
    if [[ -n $monitor ]]; then
        kill "$monitor" 2>/dev/null || true
        wait "$monitor" 2>/dev/null || true
        monitor=
    fi
}
cleanup() {
    stop_monitor
    rm -f -- "$log"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stdbuf -oL "${query[@]}" -lms 50 > "$log" &
monitor=$!
CUDA_VISIBLE_DEVICES="$gpu" "$infer" "$checkpoint" model/tokenizer.json cuda "$steps"
if ! kill -0 "$monitor" 2>/dev/null; then
    echo "显存采样进程提前退出，测量不完整。" >&2
    exit 1
fi
stop_monitor

awk -v baseline="$baseline" '
    NF == 1 && $1 ~ /^[0-9]+$/ {
        used = $1 + 0
        if (count++ == 0 || used > peak) peak = used
    }
    END {
        if (!count) {
            print "未采集到有效的显存读数。" > "/dev/stderr"
            exit 1
        }
        printf "启动前 %d MiB，整卡采样峰值 %d MiB，较启动前增加 %+d MiB\n", baseline, peak, peak - baseline
    }
' "$log"
