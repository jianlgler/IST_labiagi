
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os

from torchvision import transforms

save_check_dir = "./model"

def calc_mean_std(input, eps=1e-5):
  batch_size, channels = input.shape[:2]

  reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
  mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
  std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                    # calculate std and reshape
  return mean, std

def ada_in(content, style):
  assert content.shape[:2] == style.shape[:2] # Only first two dim, such that different image sizes is possible
  batch_size, n_channels = content.shape[:2]
  mean_content, std_content = calc_mean_std(content)
  mean_style, std_style = calc_mean_std(style)

  output = std_style*((content - mean_content) / (std_content)) + mean_style # Normalise, then modify mean and std
  return output

def Content_loss(input, target): # Content loss is a simple MSE Loss
  loss = F.mse_loss(input, target)
  return loss

def Style_loss(input, target):
  mean_loss, std_loss = 0, 0

  for input_layer, target_layer in zip(input, target): 
    mean_input_layer, std_input_layer = calc_mean_std(input_layer)
    mean_target_layer, std_target_layer = calc_mean_std(target_layer)

    mean_loss += F.mse_loss(mean_input_layer, mean_target_layer)
    std_loss += F.mse_loss(std_input_layer, std_target_layer)

  return mean_loss+std_loss

def save_state(net, iters):
  name="decoder_iter_{:d}.pth.tar".format(iters)
  state_dict = net.decoder.state_dict()
  for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
  torch.save(state_dict, os.path.join(save_check_dir, name))
  print("Checkpoint saved successfully, iters: ", iters)

def imshow(tensor, title=None):
  toPIL = transforms.ToPILImage(mode="RGB")

  image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
  image = tensor.squeeze(0)      # remove the fake batch dimension
  #image = unloader(image)
  image = toPIL(image)
  plt.imshow(image)
  if title is not None:
      plt.title(title)
  plt.pause(0.001) # pause a bit so that plots are updated