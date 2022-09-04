import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm

import os
import sys

from Utils import net, im, utils

manualSeed = 999
torch.manual_seed(manualSeed)
 
path = "./images"

sys.path.append(path)

device = ("cuda" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = learning_rate / (1.0 + learning_rate_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def training_loop(network, # StyleTransferNetwork
                  dataloader_comb, # DataLoader
                  n_epochs, # Number of Epochs
                  run_dir # Directory in which the checkpoints and tensorboard files are saved
                  ):
  
  writer = SummaryWriter(os.path.join(path, run_dir))
  # Fixed images to compare over time
  fixed_batch_style, fixed_batch_content = all_img[0]
  fixed_batch_style, fixed_batch_content =  fixed_batch_style.unsqueeze(0).to(device), fixed_batch_content.unsqueeze(0).to(device) # Move images to device

  writer.add_image("Style", torchvision.utils.make_grid(fixed_batch_style))
  writer.add_image("Content", torchvision.utils.make_grid(fixed_batch_content))

  iters = network.iters

  for epoch in range(1, n_epochs+1):
    tqdm_object = tqdm(dataloader_comb, total=len(dataloader_comb))

    for style_imgs, content_imgs in tqdm_object:
      adjust_learning_rate(optimizer, iters)
      style_imgs = style_imgs.to(device)
      content_imgs = content_imgs.to(device)

      content_loss, style_loss = network(style_imgs, content_imgs)

      content_loss = content_loss * content_w
      style_loss = style_loss * style_w
      loss_comb = style_loss + content_loss

      optimizer.zero_grad()
      loss_comb.backward()
      optimizer.step()

      # Update status bar, add Loss, add Images
      tqdm_object.set_postfix_str("Combined Loss: {:.3f}, Style Loss: {:.3f}, Content Loss: {:.3f}".format(
                                  loss_comb.item()*100, style_loss.item()*100, content_loss.item()*100))
    
      if iters % 25 == 0:
        writer.add_scalar("Combined Loss", loss_comb*1000, iters)
        writer.add_scalar("Style Loss", style_loss*1000, iters)
        writer.add_scalar("Content Loss", content_loss*1000, iters)

      if (iters+1) % 4000 == 1:
          utils.save_state(network, iters)
          writer.close()
          writer = SummaryWriter(os.path.join(path, run_dir))

      iters += 1
      print(iters)


imsize = 512
loader = transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(256),  transforms.ToTensor()])  # transform it into a torch tensor

pathStyleImages = "./images/wikiart"
pathContentImages = "./images/coco"

all_img = im.Images(pathStyleImages, pathContentImages, transforms=loader)

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load("./model/vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

learning_rate = 1e-4
learning_rate_decay = 5e-5
content_w = 1.0
style_w = 10.0

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=learning_rate)

max_iter = 160000

dataloader_comb = DataLoader(all_img, batch_size=5, shuffle=True, num_workers=0, drop_last=True)
n_epochs = 5
run_dir = "../runs/Run 1" 

training_loop(network, dataloader_comb, n_epochs, run_dir)