#!/bin/sh

set -e

mkdir -p training/checkpoints

DATASET=horse2zebra
FILE=${DATASET}.zip

if [ ! -e training/$FILE ] ; then
  URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE
  wget $URL -O training/$FILE
fi

if [ ! -d training/$DATASET ] ; then
  cd training
  echo "extracting the dataset..."
  unzip -q $FILE
  echo "done"
  cd ..
fi

#optirun -b primus python3 ./train.py \
python3 ./train.py \
  --train_log_dir="./training/checkpoints/default" \
  --image_set_x_file_pattern="./training/$DATASET/trainA/*.jpg" \
  --image_set_y_file_pattern="./training/$DATASET/trainB/*.jpg" \
  --patch_size=256 \
  --generator_lr=0.0001
