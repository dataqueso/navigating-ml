---
layout: post
title:  "1. Level 3 Setup"
categories: ml data-science tools
img: seventh.jpg
categories: [ds, ml]
---

# Setup with GPU

These are some highly rated suggestions.  A CPU machine can not perform matrix addition and multiplication as quickly, a central theme in deep learning, thus is not a viable option for training a complex neural network like ResNet, from scratch. 

## Getting your Environment Set Up

### Hardware

A recent study states the following regarding GPUs:

> _Their computational power exceeds that of the CPU by orders of magnitude:
while a conventional CPU has a peak performance of around 20 Gigaflops, a NVIDIA GeForce
8800 Ultra reaches theoretically 500 Gigaflops._
<p align="right">J.A. van Meel et al. arXiv. Published:  November 8, 2018</p>
<p align="right">https://arxiv.org/PS_cache/arxiv/pdf/0709/0709.3225v1.pdf</p>
<br>

Thus accepted options for training deep learning models are GPU(s) or TPU(s).  Other hardware options include [ASIC chips](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit).

#### Computer-Laptop

Anything with a NVIDIA GTX 1060 (GPU) is good if you travel a lot or cannot necessarily depend on wifi.

  * E.g.:  https://www.razerzone.com/gaming-systems/razer-blade-pro - plan to get a 1060 w/ 256ssd + 2tb spinner
  * GTX 1080 (extra $2k; only get if this is your primary machine and you travel a lot/have spotty wifi).

#### Computer-Desktop

Any gaming desktop with a min GTX 1080. 

1. Alt 1: Custom built.  If you go this route -> Add up your GPU ram, multiply by 2 for your min RAM.  Get a CPU w/ 48 lanes (so you can go 2 GPUs later).
2. Alt 2: [https://lambdal.com/products/quad](https://lambdal.com/products/quad) (this is probably over kill honestly, and your electricity bill might double.  Will make a great space heater)

#### Computer-Remote

Use Azure and set up a jupyter notebook.  This takes more learning and understanding but is the cheapest getting started option.

It's suggested to provision a new NC-6 (or NC12) (v3) DSVM on Ubuntu, updating everything, installing the packages, adding a 1TB HDD data disk ([add disk to the DSVM](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/attach-disk-portal)) and kicking off a password protected Jupyter Notebook in a tmux session.  DO NOT put sensitive data on this.  It has open ports, admin rights to the system, is password protected only and it's not believed to use SSL unless you set up a certificate.  There are other more secure options, though they are more advanced and not covered in this getting started.

#### IoT-Test_Device

Raspberry Pi v3, Nvidia TX-1, Nvidia TX-2 or Xavier if you can spend some cash.  If you are feeling adventurous get a few arduinos as well.

### Software

  1. Docker for Mac or Docker for Windows (avoid Toolbox)
  2. [Anaconda](https://www.anaconda.com/download/)
  3. If you have a GPU
     1. Go to [Cuda Downloads](https://developer.nvidia.com/cuda-downloads)
	 2. Join Nvidia Developer program
	 3. Update your GPU drivers
	 4. Install CUDA & cuDNN
  4. Install your packages

Good Idea:

1. StackOverflow account