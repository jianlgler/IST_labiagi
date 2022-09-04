import argparse
from configparser import Interpolation
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt

import os
import sys

from Utils import net_re, utils

def imshow(tensor, title=None):
  image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
  image = tensor.squeeze(0)      # remove the fake batch dimension
  #image = unloader(image)
  image = toPIL(image)
  plt.imshow(image)
  if title is not None:
      plt.title(title)
  plt.pause(0.001) # pause a bit so that plots are updated


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = utils.ada_in(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = utils.ada_in(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


# Path 
path = ".\images"
sys.path.append(path)

style_img = "asheville.jpg"
content_img = "cornell.jpg"
style_dir = "style"
content_dir = "content"
style_path, content_path = os.path.join(path, style_dir, style_img), os.path.join(path, content_dir, content_img)

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#if args.style:
#    style_paths = args.style.split(',')
#    if len(style_paths) == 1:
#        style_paths = [Path(args.style)]
#    else:
#        do_interpolation = True
#        assert (args.style_interpolation_weights != ''), \
#            'Please specify interpolation weights'
#        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
#        interpolation_weights = [w / sum(weights) for w in weights]
#else:
#    style_dir = Path(args.style_dir)
#    style_paths = [f for f in style_dir.glob('*')]

decoder = net_re.decoder_re
vgg = net_re.vgg_re

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load("./model/decoder.pth"))
vgg.load_state_dict(torch.load("./model/vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

content_tf = transform
style_tf = transform
toPIL = transforms.ToPILImage(mode="RGB")

alpha = 0.5
interpolation_weights = 1,1,1,1

#for content_path in content_paths:
    # process one content and one style
#    for style_path in style_paths:
content = content_tf(Image.open(content_path))
style = style_tf(Image.open(style_path))
#if args.preserve_color:
#    style = coral(style, content)
style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)
with torch.no_grad():
    output = style_transfer(vgg, decoder, content, style,
                            alpha)
output = output.cpu()
output_name = '{:s}_stylized_{:s}_{:s}'.format(os.path.splitext(content_img)[0], 
                                               os.path.splitext(style_img)[0], str(alpha))
output_name += ".jpg"

output_dir = "./images/output"
save_image(output, os.path.join(output_dir, output_name))
