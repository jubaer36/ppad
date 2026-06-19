#!/bin/bash
# run.sh - Train and evaluate PPAD on a dataset.
# Usage:
#   ./run.sh --dataset mvtec --data_path /path/to/mvtec
#   ./run.sh --dataset visa --data_path /path/to/visa --epochs 50

# Default parameters
DATASET="mvtec"
DATA_PATH=""
CATEGORY="all"
EPOCHS=10
BATCH_SIZE=8
ENCODER="dinov2_vits14"
CKPT_DIR="checkpoints"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --data_path) DATA_PATH="$2"; shift ;;
        --category) CATEGORY="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --encoder) ENCODER="$2"; shift ;;
        --ckpt_dir) CKPT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATA_PATH" ]; then
    echo "Error: --data_path is required."
    echo "Usage: ./run.sh --dataset <mvtec|visa> --data_path <path> [options]"
    exit 1
fi

echo "======================================================================"
echo "PPAD Pipeline: Dataset=$DATASET, Path=$DATA_PATH, Category=$CATEGORY"
echo "======================================================================"

# 1. Train
echo -e "\n>>> Starting Training..."
python train.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --category "$CATEGORY" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --encoder "$ENCODER" \
    --output_dir "$CKPT_DIR"

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

# 2. Evaluate
echo -e "\n>>> Starting Evaluation..."
python evaluate.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --category "$CATEGORY" \
    --ckpt_dir "$CKPT_DIR"

if [ $? -ne 0 ]; then
    echo "Evaluation failed!"
    exit 1
fi

echo -e "\n>>> PPAD Pipeline completed successfully!"
