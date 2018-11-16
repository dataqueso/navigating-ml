---
layout: post
title:  "1. Level 3 Setup"
categories: ml data-science tools
img: seventh.jpg
categories: [ds, ml]
---

# Setup

You'll be doing most things locally in this Advanced Challenge.

## Getting your Environment Set Up

### Hardware

These are some highly rated suggestions.  You may of course try out the Practice section on a CPU machine.  There's a cloud option below as well.

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

It's suggested to provision a new NC-6 (or NC12) DSVM on Ubuntu, updating everything, installing the packages, adding a 1TB data disk and kicking off a password protected Jupyter Notebook in a tmux session.  DO NOT put sensitive data on this.  It has open ports, admin rights to the system, is password protected only and it's not believed to use SSL unless you set up a certificate.  There are other more secure options, though they are more advanced and not covered in this getting started.

#### IoT-Test_Device

Raspberry Pi v3, Nvidia TX-1 or Nvidia TX-2.  If you are feeling adventurous get a few arduinos as well.

### Software

  1. Docker for Mac or Docker for Windows (avoid Toolbox)
  2. [Anaconda](https://www.anaconda.com/download/)
  3. If you have a GPU
     1. Go to [Cuda Downloads](https://developer.nvidia.com/cuda-downloads)
	 2. Join Nvidia Developer program
	 3. Update your GPU drivers
	 4. Install Cuda & Cudnn
  4. Install your packages

Good Idea:

1. StackOverflow account