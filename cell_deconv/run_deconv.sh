#!/bin/bash
# Multi-GPU parallel cell2location deconvolution

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/batch_inference.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# GPUs to use (adjust as needed)
GPUS=(0 1 2)
TOTAL_GPUS=${#GPUS[@]}

echo "Starting ${TOTAL_GPUS}-GPU parallel deconvolution..."
echo "Total tasks: 127 slides (BC:68 + HER2:36 + Kidney:23)"
echo "~$((127 / TOTAL_GPUS)) slides per GPU"
echo ""

for idx in ${!GPUS[@]}; do
    gpu_id=${GPUS[$idx]}
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT" \
        --gpu $idx --total_gpus $TOTAL_GPUS \
        2>&1 | tee "$LOG_DIR/deconv_gpu${gpu_id}.log" &
    echo "GPU $gpu_id started -> logs/deconv_gpu${gpu_id}.log"
done

echo ""
echo "All GPUs started!"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/deconv_gpu0.log"
echo ""
echo "Check completed:"
echo "  ls $SCRIPT_DIR/deconv_results_hvg2000/*/*/cell_proportion.csv 2>/dev/null | wc -l"

wait
echo "All GPUs finished!"
