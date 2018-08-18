#!/bin/sh

set -e

TRAIN_LOG_DIR=./training/checkpoints/default
MODEL=$(tail --lines=1 $TRAIN_LOG_DIR/checkpoint | \
        grep -oP '"\K[^"\047]+(?=["\047])')

python3 ./inference.py \
  --model=$TRAIN_LOG_DIR/$MODEL \
  --resnet=6 \
  --video=../samples/muybridge.mp4
