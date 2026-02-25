#!/bin/bash
# Run HER2 + Kidney + BC training: 3 datasets x 3 mask ratios x 4 folds
# 3 GPUs (0-2), each runs one mask_ratio, datasets run sequentially
# Usage: bash run_train.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "Dual-Backbone Training"
echo "  HER2 + Kidney + BC, mask {100, 300, 500}"
echo "  GPUs: 0, 1, 2"
echo "============================================"

# --- Round 1: HER2 ---
echo ""
echo ">>> Round 1: HER2 (3 mask ratios in parallel)"

# GPU 0: HER2 mask_100
nohup python "$SCRIPT_DIR/train.py" \
    --dataset HER2 \
    --mask_ratio 100 \
    --gpu 0 \
    > "$LOG_DIR/her2_mask100.log" 2>&1 &
PID1=$!
echo "  GPU 0: HER2 mask_100 (PID: $PID1)"

# GPU 1: HER2 mask_300
nohup python "$SCRIPT_DIR/train.py" \
    --dataset HER2 \
    --mask_ratio 300 \
    --gpu 1 \
    > "$LOG_DIR/her2_mask300.log" 2>&1 &
PID2=$!
echo "  GPU 1: HER2 mask_300 (PID: $PID2)"

# GPU 2: HER2 mask_500
nohup python "$SCRIPT_DIR/train.py" \
    --dataset HER2 \
    --mask_ratio 500 \
    --gpu 2 \
    > "$LOG_DIR/her2_mask500.log" 2>&1 &
PID3=$!
echo "  GPU 2: HER2 mask_500 (PID: $PID3)"

echo ""
echo "HER2 logs:"
echo "  tail -f $LOG_DIR/her2_mask100.log"
echo "  tail -f $LOG_DIR/her2_mask300.log"
echo "  tail -f $LOG_DIR/her2_mask500.log"
echo "PIDs: $PID1 $PID2 $PID3"

wait $PID1 $PID2 $PID3
echo ">>> HER2 all done!"

# --- Round 2: Kidney ---
echo ""
echo ">>> Round 2: Kidney (3 mask ratios in parallel)"

# GPU 0: Kidney mask_100
nohup python "$SCRIPT_DIR/train.py" \
    --dataset Kidney \
    --mask_ratio 100 \
    --gpu 0 \
    > "$LOG_DIR/kidney_mask100.log" 2>&1 &
PID4=$!
echo "  GPU 0: Kidney mask_100 (PID: $PID4)"

# GPU 1: Kidney mask_300
nohup python "$SCRIPT_DIR/train.py" \
    --dataset Kidney \
    --mask_ratio 300 \
    --gpu 1 \
    > "$LOG_DIR/kidney_mask300.log" 2>&1 &
PID5=$!
echo "  GPU 1: Kidney mask_300 (PID: $PID5)"

# GPU 2: Kidney mask_500
nohup python "$SCRIPT_DIR/train.py" \
    --dataset Kidney \
    --mask_ratio 500 \
    --gpu 2 \
    > "$LOG_DIR/kidney_mask500.log" 2>&1 &
PID6=$!
echo "  GPU 2: Kidney mask_500 (PID: $PID6)"

echo ""
echo "Kidney logs:"
echo "  tail -f $LOG_DIR/kidney_mask100.log"
echo "  tail -f $LOG_DIR/kidney_mask300.log"
echo "  tail -f $LOG_DIR/kidney_mask500.log"
echo "PIDs: $PID4 $PID5 $PID6"

wait $PID4 $PID5 $PID6
echo ">>> Kidney all done!"

# --- Round 3: BC ---
echo ""
echo ">>> Round 3: BC (3 mask ratios in parallel)"

# GPU 0: BC mask_100
nohup python "$SCRIPT_DIR/train.py" \
    --dataset BC \
    --mask_ratio 100 \
    --gpu 0 \
    > "$LOG_DIR/bc_mask100.log" 2>&1 &
PID7=$!
echo "  GPU 0: BC mask_100 (PID: $PID7)"

# GPU 1: BC mask_300
nohup python "$SCRIPT_DIR/train.py" \
    --dataset BC \
    --mask_ratio 300 \
    --gpu 1 \
    > "$LOG_DIR/bc_mask300.log" 2>&1 &
PID8=$!
echo "  GPU 1: BC mask_300 (PID: $PID8)"

# GPU 2: BC mask_500
nohup python "$SCRIPT_DIR/train.py" \
    --dataset BC \
    --mask_ratio 500 \
    --gpu 2 \
    > "$LOG_DIR/bc_mask500.log" 2>&1 &
PID9=$!
echo "  GPU 2: BC mask_500 (PID: $PID9)"

echo ""
echo "BC logs:"
echo "  tail -f $LOG_DIR/bc_mask100.log"
echo "  tail -f $LOG_DIR/bc_mask300.log"
echo "  tail -f $LOG_DIR/bc_mask500.log"
echo "PIDs: $PID7 $PID8 $PID9"

wait $PID7 $PID8 $PID9
echo ">>> BC all done!"

echo ""
echo "============================================"
echo "All training completed!"
echo "============================================"
