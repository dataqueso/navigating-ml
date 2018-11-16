---
layout: post
title:  "2. Level 2 Setup"
categories: ml data-science tools
img: fourth.jpg
categories: [ds, ml]
---

# Setup

## Cloud

1. (Deploy when ready if you don't have it yet) [Linux (Ubuntu) Deep Learning Virtual Machine](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-deep-learning-dsvm) Standard NC6

> Tip: place everything for this set of challenges in the same resource group to tear down together at the end.

 > TensorFlow and PyTorch can be found on all Linux and Windows Data Science Virtual Machines and Deep Learning Virtual Machines along with a long list of common data science tools - see [Docs](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/overview)

2. Ensure Jupyter notebooks are working.
    - "access the Jupyter notebook server from any host. Just navigate in the browser to `https://<VM DNS name or IP Address>:8000/`" ([Doc](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro#tools-installed-on-the-data-science-virtual-machine-for-linux) - check out "Jupyter notebook" for more)
 
## Local via Docker

Here are the instructions for a generic scientific and deep learning custom Linux VM with Jupyterhub (Running Locally):

* GitHub project - [Ref](https://github.com/michhar/custom-jupyterhub-linux-vm)

## Local Non-Docker

Here, it's assumed you know what you are doing and can set up the scientific stack.

Important additional libraries:

* Jupyter - [Ref](https://jupyter-notebook.readthedocs.io/en/stable/)
* PyTorch - [Ref](https://pytorch.org/)
 

