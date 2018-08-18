#!/bin/sh

set -e

mkdir -p build

cd build

if [ ! -e predict ] ; then
  cmake ..
  make
fi

./predict
