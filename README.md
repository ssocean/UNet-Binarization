# A UNet(or any other FCN)-based repo for segmentation, especially for binarization.

### This repo also hold the Official code of ~~[Attention-U-Net(deprecated)](https://github.com/ssocean/Attention-U-Net)~~ and order prediction in [IACC-DAR-AlphX-Code](https://github.com/ssocean/AlphX-Code-For-DAR).



## Introduction

This project aims to provide a solution for image segmentation that can be used in many fields, e.g. document binarization. This repo is simple, efficient and flexiable, you can modify anything you want. 

## Installation

Ensure that the following Python packages have been installed:

```python
pip install numpy
pip install torch
pip install torchvision
pip install opencv-python
pip install tensorboard
pip install tqdm
```

or just pip install the missing package is more than enough.

## Usage


### Prepare the data

Set `imgs_dir` and `masks_dir` to your path.

Here, `--input` is the path to the input image and `--output` is the path to where the model will write the output.

## Training the Model

If you wish to train the model or use your own dataset, follow these steps:

1. Prepare your data as requested. 

2. Navigate to the base directory in the terminal and run the following command:

`python train.py`

```
Args :

--imgs_dir: Directory of input images

--masks_dir: Directory of GT masks

--dir_checkpoint: Directory to save the checkpoints.

--input_size: Size of input images

--epoch: Number of epochs for training

--batch_size: Batch size for training

--val_percent: Percentage of validation data

--lr: Learning rate for training

--weight_decay: Weight decay factor for training

--momentum: Momentum factor for the optimizer
```

You can add those args if needed.

## Model Inference

`python infer.py`

```
Args :

--imgs_dir: The directory where input images are located.

--out_dir: The directory to save the output images after processing.

--model_pth: The path to the trained network model.

--batch_size: The number of images to process in each batch.

--patch_size: The size of each image patch to be processed, recommand 256.

--bitwise_img_size: The size of the images after bitwise operations. We recommend setting this value to as large as possible.
```

You can add those args if needed.





