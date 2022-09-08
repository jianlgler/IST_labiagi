# Through an Interactive Style transfer

> Multiple implementations of Neural Style Transfer

![alt text](https://github.com/jianlgler/IST_labiagi/blob/main/images/output/newyork_stylized_Katsushika_Hokusai_Yoshitsune_Falls_1.jpg)

Inspired by: 
  - Leon Gatys et al.'s paper: [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
  - Xun Huang et al.'s paper: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)
# Index

- [How to Start](#how-to-start)
- [How to Run](#how-to-run)
- [The Project](#the-project)
- [Author](#author)

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

# The Project

# Author 
Gianluca Trovò, Computer Engineering Student, Sapienza, Rome
