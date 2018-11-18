---
layout: post
title:  "Level 3 Practice"
categories: ml data-science tools
img: ninth.jpg
categories: [ds, ml]
---

# Advanced Deep Learning with PyTorch

In this advanced Challenge, the instructions will be a little more vague and you'll need to go figure find out much on your own, part of the learning and challenge.

_Why do this task_:  Usually, beginner tutorials around ML and neural networks begin with classifying hand-written digits from the MNIST dataset or CIFAR-10.  We are going to begin with something more challenging and much of it will be dealing with data and data formats.  This is to simulate how life will likely be in real life and it's hoped you will learn how to create machine learning models more effectively and quickly in the real world. The reason to work through the following is:

  * It will force you to read and learn from scratch.  You will learn the different label file formats, deserializers and how things compute. 
  * For for example, in energy/manufacturing you will get .png or .jpg or .tiff files and not stuff already in the perfect format. 
  * Learning this will hopefully help you understand the concept of “Data Packing”. 
  * This is not the simplest way, but it forces greater learning.

## Working with PyTorch (locally or on a DSVM/VM)

See [Level 2 Setup](/navigating-ml/level2_setup) for more instructions on how to set up an environment for this problem set.

1. Image Classification

    1. Start with the Hymenoptera insect raw data
        1. Get Data from here: [click to download](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
        2. Use `transforms` modules from `torchvision` and other libraries to:
            * Try out some data augmentation - (random vertical flip and blur the images)
        3. Make sure you also create an example for [inference](https://en.wikipedia.org/wiki/Statistical_inference).
        2. Use Scikit-learns’s confusion matrix and classification_report to generate metrics.
            1. [Scikit-learn's confusion matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
            2. [Scikit-learn's classification report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

## Want More?

4. Do the same exact exercise with CoCo: http://cocodataset.org/#home
    1. Why do you think you get bad results?
<br><br>
1.  Object Detection
    1. Use the Out of Box Faster-RCNN or YOLOv3 solution to identify new objects in images based on a model trained on CoCo

## TensorFlow

Train a CNN on the CIFAR-10 dataset as in this [Tutorial](https://www.tensorflow.org/tutorials/deep_cnn).

## Key Learnings

## Additional Help

* PyTorch forums - [Ref](https://discuss.pytorch.org/)
* StackOverflow with `pytorch` or `tensorflow` tag
* If using CNTK, you may send your questions to cntkhelp@microsoft.com
