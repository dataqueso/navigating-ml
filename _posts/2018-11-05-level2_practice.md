---
layout: post
title:  "3. Level 2 Practice"
categories: ml data-science tools
img: sixth.jpg
categories: [ds, ml]
---

# Intermediate Practice with Deep Learning

In this intermediate set of problems, you'll apply what you've learned in the Level 2 Preparation section.  You'll start exploring the CIFAR-10 dataset with an MLP and CNN along with other datsets in PyTorch and TensorFlow as extra credit.

## Adapt a Multilayer Perceptron

### Image Classification

Instructions to practice image classification with PyTorch

2. Let's dive into some code.  Open a Jupyter session or log into the Jupyter system (on your DSVM or locally)
  - In a code cell `! git clone https://github.com/rasbt/deep-learning-book.git`
  - Open this notebook:  `/code/model_zoo/pytorch_ipynb/multilayer-perceptron.ipynb`
  - Modify the notebook to work with the CIFAR-10 dataset
    * Remember you're working with RGB images instead of grayscale
  - What is the resulting average test error?  Why is this value so different from the MNIST result?  What hyperparameters can you modify to fix this?
2. Go online and find 5 png's of cats and dogs.  Reshape them and pad them to be 32x32 pixels using the Python Pillow library (see [ImageOps](http://pillow.readthedocs.io/en/3.1.x/reference/ImageOps.html)). Test the network with these, following the guidelines and lessons you learned thus far.  Now find an image of an apple and test the network with this.  What is wrong with using a food image?
3. Create a new label called "apple" and add the apple images from the [fruit dataset](http://www.vicos.si/Downloads/FIDS30), splitting into train/test and adding to previous data.  What are your findings after testing with the same apple image?

## Transfer Learning with PyTorch

Here, for ease of use and speed we'll use Transfer Learning as well.

1. Take [this](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) tutorial to work with Inception v3 as the base model (model [choices](https://pytorch.org/docs/stable/torchvision/models.html)) - write code to load the model, count the features going in to a fully connected layer (last layer of the CNN) and reset it to a Linear layer to thus unfreeze the layer for transfer learning)
2.  Using the CIFAR-10 dataset from PyTorch `datasets`, train an Inception v3 model to classify trucks and automobiles (just a two-class classifier from the 10 classes in CIFAR).


## CNNs with TensorFlow

Easy:  Run this TensorFlow script to classify a new image (this uses a pretrained Inception V3 model):

* [https://www.tensorflow.org/tutorials/image_recognition](https://www.tensorflow.org/tutorials/image_recognition)

Intermediate: Perform this TensorFlow CNN Tutorial from Google:

* [https://www.tensorflow.org/tutorials/deep_cnn](https://www.tensorflow.org/tutorials/deep_cnn)

Advanced:  Modify this MNIST CNN TensorFlow tutorial for use with the CIFAR-10 dataset:

* [http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/](http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/)

## Want More?

Check out Rodrigo Benenson's [blog](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130) to find out the best algorithm for classifying with CIFAR-10 and implement it.  May the force be with you.

## Additional Help

* PyTorch forums - [Ref](https://discuss.pytorch.org/)
* StackOverflow with `pytorch` or `tensorflow` tag
