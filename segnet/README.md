# SegNet

![enter image description here](http://mishurov.co.uk/images/github/cv_ml_probes/segnet.png)

As a base I took the implementation from here https://github.com/aizawan/segnet . Unfortunately the implemented model didn't behave as expected and I had to make some modifications. Particularly, I set new batch normalisation layers with batch renormalisation and proper op updates. Model predictions became consistent with the training mode. I've changed the convolution layers as well.

My main focus was to adapt the model using Datasets and Estimators API and evaluate it using C++ API within OpenCV environment. Also I added training and image summaries for TensorBoard for more a transparent training process.

Since the project uses Tensorflow's C++ API, Tensorflow needs to be compiled from the sources for the necessary link libraries and header files. 
