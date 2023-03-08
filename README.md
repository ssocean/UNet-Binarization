# A UNet(or any other FCN)-based repo for segmentation, especially for binarization.

# This repo also hold the Official code of ~~[Attention-U-Net(deprecated)](https://github.com/ssocean/Attention-U-Net)~~ and order prediction in [IACC-DAR-AlphX-Code](https://github.com/ssocean/AlphX-Code-For-DAR).



## Introduction

This project aims to provide a solution for image binarization that can be used in computer vision, automation, and other fields. The model uses UNet semantic segmentation, which has shown good performance in the field of binarization.

## Installation

Ensure that the following Python packages have been installed:

···
pip install numpy
pip install torch
pip install torchvision
pip install opencv-python==4.4.0.42
pip install matplotlib
pip install tqdm
···


## Usage

Navigate to the `src/` directory in the terminal and run the following command:

·python predict.py --input input_path --output output_path·


Here, `--input` is the path to the input image and `--output` is the path to where the model will write the output.

## Training the Model

If you wish to train the model or use your own dataset, follow these steps:

1. Download the [dataset](https://github.com/ssocean/UNet-Binarization/raw/main/dataset/food_data.zip) and unzip it.

2. Launch training with the following command:
·python train.py --data data_path --epochs 20 --batch_size 32 --lr 0.01 --plot_loss·

Here, `--data` points to the path of the dataset, `--epochs` is the total number of epochs you want to train for, `--batch_size` is the batch size for each iteration, and `--lr` is the learning rate. Use the `--plot_loss` parameter if you want to generate a loss chart for the model.

3. The trained model will be saved in the `./model/` directory.

## Model Prediction Examples

Below are some examples of model outputs. More can be found in the `./results/` directory.

![Example 1](./results/1.png)

![Example 2](./results/2.png)

## References

- UNet: Convolutional Networks for Biomedical Image Segmentation. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. MICCAI 2015.

## License

This project is distributed under the [MIT License](./LICENSE).


