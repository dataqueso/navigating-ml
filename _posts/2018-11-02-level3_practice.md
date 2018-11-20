---
layout: post
title:  "Level 3 Practice"
categories: ml data-science tools
img: ninth.jpg
categories: [ds, ml]
---

# Advanced Deep Learning with PyTorch

In this advanced Challenge, the instructions will be a little more vague and you'll need to go figure find out much on your own, part of the learning and challenge.

_Why do this task_:  Usually, beginner tutorials around ML and neural networks begin with classifying hand-written digits from the MNIST dataset or CIFAR-10.  We are going to begin with something more challenging and much of it will be dealing with data, labeling and data formats.  This is to simulate how life will likely be in real life and it's hoped you will learn how to create machine learning models more effectively and quickly in the real world. The reason to work through the following is:

  * It will force you to read and learn from scratch.  You will learn the different label file formats, deserializers and how things compute. 
  * For for example, in energy/manufacturing you will get .png or .jpg or .tiff files and not stuff already in the perfect format.  You may even get video data.
  * Learning this will help you understand the concept of “Data Packing”. 
  * This is not the simplest way, but it forces greater learning.

## Working with PyTorch on More Complex Data

See [Setup](/navigating-ml/setup) for more instructions on how to set up an environment for this problem set.

### Image Classification

Start with raw video of fish swimming at a video trap in the northern territories of Australia.

1. Download the video sample from here: https://github.com/Azadehkhojandi/FindingFishwithTensorflowObjectdetectionAPI/blob/master/fish_detection/fishdata/Videos/video1.mp4
2. Separate the video input into individual frames
3. Create a classifier to help decide if a frame has a fish or not.  Use transfer learning.
2. Use `transforms` modules from `torchvision` and other libraries to:
    * Try out some data augmentation - (e.g. random vertical flip and blur the images)
3. Make sure you also create an example for [inference](https://en.wikipedia.org/wiki/Statistical_inference).
2. Use Scikit-learns’s confusion matrix and classification_report to generate metrics.
    1. [Scikit-learn's confusion matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
    2. [Scikit-learn's classification report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

### Object Detection

Use the Out of Box Faster-RCNN or YOLOv3 solution to identify fish in frames (you will need to label with bounding boxes - a good tool is VoTT or the VGG Image Annotator)

## Additional Help

* PyTorch forums - [Ref](https://discuss.pytorch.org/)
* StackOverflow with `pytorch` tag
