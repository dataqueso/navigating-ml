---
layout: post
title: 3. Level 1 Practice
img: third.jpg
---

# Going from Prebuilt ML Tools to First Custom ML

It is recommended that you have completed the [Leve 1 Preparation](level1_prep).

In this Beginner Challenge you'll learn about basic ML and neural networks hands-on with Jupyter notebooks and Python.  You'll be introduced to scikit-learn, CNTK, and TensorFlow as Python packages commonly used in data manipulation and data science.  

Here and throughout these practice exercises you'll work with the following image datasets: the fruit FIDS30 dataset, the Kaggle Fashion MNIST dataset and the CIFAR-10 (tiny images) dataset.

## Custom Vision (free from Microsoft)

Download the [fruit dataset](http://www.vicos.si/Downloads/FIDS30) and build a fruit image classifier with two fruit classes using [https://customvision.ai/](https://customvision.ai/).

After you have done some training above, create a Python script to "pixel-normalize" the images prior to training the model and retrain to see your new Precision and Recall.

![Precision and recall](http://nlpforhackers.io/wp-content/uploads/2017/01/Precision-Recall.png)

> Some defintions.  **Precision**:  if a tag is precicted by your classifier, how likely is it that it is right?  **Recall**:  out of the tags that should be classified as right, what percentage did your classifier correctly find?


## First Custom ML (Open Source Tools)

For these two problems, it is recommended to go through the code from the original source line by line in whatever fashion you see fit so that you really understand what is happening.

> TIPS:  Place all imports at the top of the notebook.  Call the training data something consistent thoughout all of your work (X_train -> training data, y_train -> labels, X_test -> test data...).

### Image Classification

Create a Python program to classify images from Fashion MNIST Dataset (get [here](https://www.kaggle.com/zalando-research/fashionmnist/data)) leveraging code samples from the Python Data Science Handbook - [Ref](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html#Application:-Exploring-Hand-written-Digits).  Refer to Chapter 2 and 3 of this Handbook for information on data manipulation in Python if not already familiar.

Do this in a Jupyter notebook (any service or locally) - recall you learned about this tool the [Setup Section](level1_setup).  

It might help to examine the existing data format for `sklearn.datasets.load_digits` so that you can convert into that format to utilize the algorithms in `sklearn` (scikit-learn).  

- What did you find?  Which fashion item has the best accuracy, which the worst?  Why do you think that is?  Is there a way you could imagine improving this model?
- Try a different model
- Scale the images (in `sklearn`) and check the accuracy of the model again.  Did it improve or worsen?



### Object Detection

> _In the real world, data is rarely so uniform and simple pixels will not be suitable: this has led to a large literature on feature extraction methods for image data._

Create a Python program to detect cats in 2D images by leveraging code samples from the Python Data Science Handbook - [Ref](https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html).  Refer to Chapter 2 and 3 of this Handbook for information on data manipulation in Python if not already familiar.

Do this in a Jupyter notebook (any service or locally) - recall you learned about this tool the [Setup Section](level1_setup).  

It might help to examine the existing data format for `sklearn.datasets.fetch_lfw_people` so that you can convert into that format to utilize the algorithms in `sklearn` (scikit-learn) to create this detector.  

- What other confounding factors are there for images other than illumination, you think?
- Plot the original image along with the `skimage.rgb2gray` version and the HOG representation.  See how this works in `matplotlib`.  What does `skimage.rgb2gray` actually do?
- Can you scale the new image of the astronaut with the `PIL` module instead?  This module is very powerful and good to know about (as well as `opencv`)?
- Try out the model on the entire test image instead.  What do you find out?
- Try using sliding windows with a variety of sizes (aspect ratios).  What do you find out?
- Read in a new image that contains a face and on one that does not and try your model on that.
- Augment the data to expand the training and test datasets (e.g. use a library like `imgaug`) and retrain and test.
- **Extra credit**:  Implement Non-Maximum Suppression in Python to find the single best bounding box of a group of bounding boxes as are found above.  Apply this to the astronaut image.


## Basic Neural Nets

The purpose of the Basic Neural Nets exercises are to familiarize you with how a simple artificial neuron works and then set of a few neurons to form a network (artificial neural network) - all from the ground-up - this knowledge will serve you well.  It will really get to the core of neural nets and give you a perfect "from scratch" introduction (the code template already exists around the infamous iris dataset, you'll just make it work with some fashionable images).  If we want to get to know deep neural nets, why not dive in deep to start!

1.  Use [Azure Notebooks](https://notebooks.azure.com) for this tutorial.  Fire up a blank Python 3.5 Jupyter notebook for this.
2. Get the following sample image dataset loaded in your Jupyter notebook: [train and test Fashion MNIST Dataset (Source: Kaggle)](https://www.kaggle.com/zalando-research/fashionmnist/data)
3. Adapt a from-scratch Perceptron as in this [Jupyter notebook](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch02/ch02.ipynb) to train and test on the image dataset.
    - Use the URL option when opening up a new notebook in Azure Notebooks
    - Or, download by right clicking on "Raw" and "Save link as..."
    - Re-implement the Perceptron with `sklearn` (scikit-learn)
4. Adapt a from-scratch Multilayer Perceptron (MLP) as in this [Jupyter notebook](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch12/ch12.ipynb)
    - Use the URL option when opening up a new notebook in Azure Notebooks
    - Or, download by right clicking on "Raw" and "Save link as..."
    - Re-implement the MLP with `sklearn`

## Moving On

Now it is time to move on to Level 2 Preparation.

## Additional Help

- StackOverflow with `sklearn`, `jupyter`
- For Custom Vision you can email customvisionteam@microsoft.com.
