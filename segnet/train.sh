#!/bin/sh

set -e

mkdir -p training/checkpoints

DATASET=CamVid

if [ ! -d training/$DATASET ] ; then
  git clone --single-branch --depth 1 \
    https://github.com/alexgkendall/SegNet-Tutorial.git
  mv SegNet-Tutorial/$DATASET training/
  rm -rf SegNet-Tutorial
fi

#CUDA_VISIBLE_DEVICES="" python train.py
python train.py
