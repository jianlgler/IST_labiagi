# Through an Interactive Style transfer

> Multiple implementations of Neural Style Transfer

![alt text](https://github.com/jianlgler/IST_labiagi/blob/main/images/result.png)

Inspired by: 
  - Leon Gatys et al.'s paper: [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
  - Xun Huang et al.'s paper: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)
# Index

- [How to Start](#how-to-start)
- [How to Run](#how-to-run)
- [Features](#features)
- [Author](#author)
- [References](#references)

# How to Start

If you are running the ipynb files you do not have to install no dependencies. 
The neural_style_transfer.ipynb file run perfectly and you don't have to setup anything.
However for PyTorch_AdaIN.ipynb you will need to adjust paths and folders for the code to be runnable, otherwise it will throw errors. 
Note that I use the colab.drive library to import what I need (utils, input images and the network), but alternatevely you could clone this whole repository and fix paths to make it work (as an example). In any case, remember to download [the modified VGG-19 network](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view). 

## Depencencies
- ipywidgets==8.0.1
- matplotlib==3.5.3
- numpy==1.23.1
- Pillow==9.2.0
- torch==1.12.1+cu116
- torchvision==0.13.1+cu116
- tqdm==4.64.0

You can install them all by running 
```js
pip install -r requirements.txt
```

# How to Run

## Leon Gatys e al.

For the Leon Gatys e al.'s model you don't have to do anything but run the code in colab.

## Xun Huang et al.

run the following cmd: 
```js
git clone https://github.com/jianlgler/IST_labiagi.git
```

### Training
The Network is trained using both WikiArt (style) and Coco (content) datasets. To manually train the network just run train.py. 
- [Wikiart](https://archive.org/details/wikiart-dataset)
- [2017 Coco](http://images.cocodataset.org/zips/train2017.zip)

### Testing
At the moment there is no executable avaible, what I suggest is to open the folder with a Python IDE (VS Code, Atom...) and run the test.py.
To control the output and change the input images edit the code variables (alpha, color, image names and paths).

# Features

## Content/Style tradeoff

It is possible to control style's impact on the content image through a paramether used for instance normalization.

![alt text](https://github.com/jianlgler/IST_labiagi/blob/main/images/result_alpha.png)

## Preserve color

Preserve the original content image color.

![alt text](https://github.com/jianlgler/IST_labiagi/blob/main/images/result_preserve_color.png)

This feature comes from [Xun Huang's Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style) too.

# Author 
Gianluca Trovò, Computer Engineering Student, Sapienza, Rome

# References
- [1]: [Leon Gatys' Original implementation in Python](https://github.com/leongatys/PytorchNeuralStyleTransfer)
- [2]: [Xun Huang's Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
- [3]: Leon Gatys et al.'s paper: [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
- [4]: Xun Huang et al.'s paper: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)
