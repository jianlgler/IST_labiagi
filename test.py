import argparse
from configparser import Interpolation
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import os
import sys

from Utils import net, utils

# Path 
path = ".\images"
sys.path.append(path)

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


style_img = "Katsushika_Hokusai_Yoshitsune_Falls.jpg"
content_img = "chicago.jpg"
style_dir = "style"
content_dir = "content"
style_path, content_path = os.path.join(path, style_dir, style_img), os.path.join(path, content_dir, content_img)

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

decoder = net.decoder
vgg = net.vgg

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

alpha = 1
interpolation_weights = 1,1,1,1

content = content_tf(Image.open(content_path))
style = style_tf(Image.open(style_path))

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