# CycleGAN

![zebra](http://mishurov.co.uk/images/github/cv_ml_probes/cyclegan.png)

A TensorFlow Research model https://github.com/tensorflow/models/tree/master/research/gan/cyclegan

Using TensorFlow Datasets with an unspecified batch size in the input_fn and 6 resnet blocks I've managed to cram a model for 256x256 images for training into 2Gb RAM of the GeForce 940MX on my laptop. Additionally I included random flips and random crops in order to augment the dataset. Nonetheless the training requires quite a bit for more or less bearable results.

For evaluation and visualisation I use the OpenCV Python 3 binding. In theory it would work with TensorFlow and OpenCV from PyPI but one needs to edit the OpenCV imports in the corresponding python file.

"train.sh" downloads the dataset, sets up and starts training.

"inference.sh" is an example of evaluating the generator with the video file from the "samples" directory. Infinitely great respect and thank to Botany Bay Pictures Films Incfor for the colourised footage.
