#!/bin/sh

set -e

TRAINER=build/train

if [ ! -e $TRAINER ] ; then
  mkdir -p build
  cd build
  cmake ..
  make
  cd ..
fi

mkdir -p training

# https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip

if [ ! -e training/lfpw.zip ] ; then
  echo 'Download manually https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip'
  echo 'because it requires a confirmation. Put it into the "training" directory.'
  echo 'And rerun the script.'
  exit 0
fi

cd training

if [ ! -d trainset ] ; then
  echo "extracting the dataset..."
  unzip -q lfpw.zip
  echo "done"
fi

ls trainset/*.pts > annotations.txt
ls trainset/*.png > images.txt

cp ../../deps/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml .
cp ../../deps/src/opencv_contrib/modules/face/samples/sample_config_file.xml .


../$TRAINER --config="sample_config_file.xml" \
  --face_cascade="haarcascade_frontalface_alt2.xml" \
  --annotations="annotations.txt"  --images="images.txt" \
  --model="trained_model.dat"



