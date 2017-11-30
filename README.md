EMNIST
=====

![](https://raw.githubusercontent.com/Coopss/EMNIST/master/static/preview.gif)

Developed by @coopss

##### Description

This project was intended to explore the properties of convolution neural networks (CNN) and see how they compare to recurrent convolution neural networks (RCNN). This was inspired by a [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_004.pdf "Recurrent Convolutional Neural Network for Object Recognition") I read that details the effectiveness of RCNNs in object recognition as they perform or even out perform their CNN counterparts with fewer parameters. Aside from exploring CNN/RCNN effectiveness, I built a simple interface to test the more challenging [EMNIST dataset](https://arxiv.org/abs/1702.05373 "EMNIST: an extension of MNIST to handwritten letters") dataset (as opposed to the [MNIST dataset](http://yann.lecun.com/exdb/mnist/ "THE MNIST DATABASE of handwritten digits"))

##### Current Implementation
  * Multistack CNN
  * Web-applet testing environment
    * Touch screen compatible
    * Works best when letter takes up a good portion of the canvas
  * Read in .mat file
  * Currently training on the [byclass dataset](https://cloudstor.aarnet.edu.au/plus/index.php/s/7YXcasTXp727EqB/download) (*direct download link*)
    * See [paper](https://arxiv.org/abs/1702.05373 "EMNIST: an extension of MNIST to handwritten letters") for more info

##### Todo
  * Update gif with new webapp
  * Train more models
    * RCNN
    * Optimize hyperparameters
    * Add a noise (gaussian or likewise) layer to input in an attempt to boost accuracy
  * Move webapp to a host service like PythonAnywhere

## Environment

#### Anaconda: Python 3.5.3
  * Tensorflow or tensorflow-gpu (See [here](https://www.tensorflow.org/install/ "Installing TensorFlow") for more info)
  * Keras
  * Flask
  * Numpy
  * Scipy

  Note: All dependencies for current build can be found in dependencies.txt

## Usage
#### [training.py](https://github.com/Coopss/EMNIST/blob/master/training.py)
A training program for classifying the EMNIST dataset

    usage: training.py [-h] --file [--width WIDTH] [--height HEIGHT] [--max MAX] [--epochs EPOCHS] [--verbose]

##### Required Arguments:

    -f FILE, --file FILE  Path .mat file data

##### Optional Arguments

    -h, --help            show this help message and exit
    --width WIDTH         Width of the images
    --height HEIGHT       Height of the images
    --max MAX             Max amount of data to use
    --epochs EPOCHS       Number of epochs to train on
    --verbose         Enables verbose printing

#### [server.py](https://github.com/Coopss/EMNIST/blob/master/server.py)
A webapp for testing models generated from [training.py](https://github.com/Coopss/EMNIST/blob/master/training.py) on the EMNIST dataset

    usage: server.py [-h] [--bin BIN] [--host HOST] [--port PORT]

##### Optional Arguments:

    -h, --help   show this help message and exit
    --bin BIN    Directory to the bin containing the model yaml and model h5 files
    --host HOST  The host to run the flask server on
    --port PORT  The port to run the flask server on
