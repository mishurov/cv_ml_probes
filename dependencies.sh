#!/bin/bash

# sudo apt-get install swig

set -e


ENABLE_TF_GPU=1
CUDA_PATH=/usr/local/cuda
CUDA_COMPUTE=5.0


mkdir -p deps/src
mkdir -p deps/opencv

cd deps/src

if [ ! -d opencv ] ; then
  git clone -b "3.4.1" --single-branch --depth 1 \
    https://github.com/opencv/opencv.git
fi


if [ ! -d opencv_contrib ] ; then
  git clone -b "3.4.1" --single-branch --depth 1 \
    https://github.com/opencv/opencv_contrib.git
fi


if [ ! -d ../opencv ] ; then
  mkdir -p opencv_gen
  cd opencv_gen

  cmake \
    -D WITH_CUDA=0 \
    -D WITH_OPENCL=0 \
    -D BUILD_JAVA=0 \
    -D WITH_PROTOBUF=1 \
    -D WITH_OPENGL=1 \
    -D WITH_QT=0 \
    -D WITH_GTK_2_X=1 \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
    -D CMAKE_INSTALL_PREFIX=../../opencv \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
    ../opencv/

  make -j4
  make install

  # in case of direct download
  # https://raw.githubusercontent.com/opencv/opencv_3rdparty/8afa57abc8229d611c4937165d20e2a2d9fc5a12/face_landmark_model.dat
  cp share/OpenCV/testdata/cv/face/face_landmark_model.dat ../../../face/

  cd ..
fi


if [ ! -d tensorflow ] ; then
  git clone \
    -b "v1.10.0" --single-branch --depth 1 \
    --recursive https://github.com/tensorflow/tensorflow.git
fi


BAZEL_VER=0.15.2
BAZEL_INSTALLER=bazel-${BAZEL_VER}-installer-linux-x86_64.sh

if [ ! -e $BAZEL_INSTALLER ] ; then
  wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VER/$BAZEL_INSTALLER
  chmod +x $BAZEL_INSTALLER
  ./$BAZEL_INSTALLER --user
fi


if [ ! -d tensorflow/tensorflow_pkg ] ; then
  export PATH="$PATH:$HOME/bin"
  cd tensorflow
  TF_ROOT=$(pwd)

  export PYTHON_BIN_PATH=$(which python3)
  export PYTHON_LIB_PATH=$($PYTHON_BIN_PATH -c \
                           'import site; print(site.getsitepackages()[1])')
  export PYTHONPATH=${TF_ROOT}/lib
  export PYTHON_ARG=${TF_ROOT}/lib

  export TF_NEED_AWS=0
  export TF_NEED_GCP=0
  export TF_NEED_HDFS=0
  export TF_NEED_OPENCL=0
  export TF_NEED_JEMALLOC=1
  export TF_ENABLE_XLA=0
  export TF_NEED_VERBS=0
  export TF_CUDA_CLANG=0
  export TF_NEED_MKL=0
  export TF_DOWNLOAD_MKL=0
  export TF_NEED_MPI=0
  export TF_NEED_S3=0
  export TF_NEED_KAFKA=0
  export TF_NEED_GDR=0
  export TF_NEED_OPENCL_SYCL=0
  export TF_NEED_TENSORRT=0
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_NEED_GCP=0
  export TF_NEED_CUDA=$ENABLE_TF_GPU

  if [ $ENABLE_TF_GPU == 1 ] ; then
    export CUDA_TOOLKIT_PATH=$CUDA_PATH
    export CUDNN_INSTALL_PATH=$CUDA_PATH
    export NCCL_INSTALL_PATH=$CUDA_PATH

    get_def () {
      def=$1
      path=$2
      echo $(sed -n "s/^#define $def\s*\(.*\).*/\1/p" $path)
    }
    export TF_CUDA_VERSION=$($CUDA_TOOLKIT_PATH/bin/nvcc --version | \
                             sed -n 's/^.*release \(.*\),.*/\1/p')
    export TF_CUDA_COMPUTE_CAPABILITIES=$CUDA_COMPUTE
    export TF_CUDNN_VERSION=$(get_def CUDNN_MAJOR \
                              $CUDNN_INSTALL_PATH/include/cudnn.h)

    NCCL_MAJOR=$(get_def NCCL_MAJOR $CUDNN_INSTALL_PATH/include/nccl.h)
    NCCL_MINOR=$(get_def NCCL_MINOR $CUDNN_INSTALL_PATH/include/nccl.h)
    export TF_NCCL_VERSION=${NCCL_MAJOR}.${NCCL_MINOR}
  fi

  export GCC_HOST_COMPILER_PATH=$(which gcc)
  export CC_OPT_FLAGS="-march=native"

  #bazel clean

  ./configure

  bazel build --config=opt //tensorflow:libtensorflow_cc.so

  mkdir -p ../../tensorflow/lib
  cp bazel-bin/tensorflow/libtensorflow_cc.so \
     bazel-bin/tensorflow/libtensorflow_framework.so \
    ../../tensorflow/lib/
  chmod 644 ../../tensorflow/lib/*

  bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

  if [ $ENABLE_TF_GPU == 1 ] ; then GPU_FLAG="--gpu" ; else GPU_FLAG="" ; fi

  bazel-bin/tensorflow/tools/pip_package/build_pip_package $GPU_FLAG tensorflow_pkg

  cd ..
fi


if [ ! -d ../cvml ] ; then
  virtualenv -p python3 ../cvml
  deactivate || true
  source ../cvml/bin/activate

  #scipy==1.1.0
  pip install gym==0.10.5
  pip install 'gym[box2d]'

  pip install tensorflow/tensorflow_pkg/tensorflow*.whl

  # copy tensorflow headers from the python package
  SITE_PACKAGES=$(python -c \
    "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
  cp -r $SITE_PACKAGES/tensorflow/include ../tensorflow

  # copy tensorflow/cc headers from the repo
  SRC_DIR=$(pwd)
  cd $SRC_DIR/tensorflow/tensorflow
  find cc -name \*.h -exec cp --parents {} \
    $SRC_DIR/../tensorflow/include/tensorflow/ \;

  # copy tensorflow/cc/ops headers from the bazel build
  cd $SRC_DIR/tensorflow/bazel-genfiles/tensorflow
  find cc -name \*.h -exec cp --parents {} \
    $SRC_DIR/../tensorflow/include/tensorflow/ \;

  # fix mode bits
  find $SRC_DIR/../tensorflow/include/ -type d -print0 | xargs -0 chmod 0755
  find $SRC_DIR/../tensorflow/include/ -type f -print0 | xargs -0 chmod 0644

  cd $SRC_DIR
fi

