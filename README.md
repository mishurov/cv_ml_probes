# Computer Vision and Machine Learning probes

This repository contains 5 small projects made using OpenCV and TensorFlow.

The projects are in the following directories
 - **face** : Facial landmark detection using OpenCV Facemark API
 - **ar** : Feature detection and matching using SIFT
 - **cyclegan** : Modified tensorflow research model of CycleGAN
 - **segnet** : Implementation of SegNet
 - **rl** : Asynchronous Advantage Actor-Critic for BipedalWalker-v2

The projects were developed on Debian Stretch, there's the script "dependencies.sh" which compiles from the sources OpenCV 3.4.1 and Tensorflow 1.10. There're some build dependencies, since they were installed on my system, I don't know for sure what they are and I don't call any "apt-get install" commands in the script. Necessary things I guess can be figured out after googling CMake errors during generating projects.

TensorFlow from the sources is needed mostly for C++ API which I was used for evaluating SegNet. For the other projects, I guess, TensorFlow from PyPI would be sufficient. OpenCV from the sources is needed because of the contrib modules which contain Facemark API. The other projects don't use the contrib and also probably would run with the distributed OpenCV 3.4.1 binaries, headers and the Python 3 bindings.

