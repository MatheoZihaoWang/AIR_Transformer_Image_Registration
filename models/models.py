# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.WarpST import WarpST
from models.ops import *
import matplotlib.pyplot as plt
import numpy as np
from models.torch_similarity.functional import normalized_cross_correlation
from torch.autograd import Variable
from utils.constrains import Bend_Penalty, LCC, Grad, RobustLOSS

def plot(im):
  im = np.array(im.tolist())
  plt.imshow(im, cmap='gray', vmin=0, vmax=1)
  plt.show()
  return None

class Warper2d(nn.Module):
  def __init__(self, img_size):
    super(Warper2d, self).__init__()
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
#        img_src: [B, 1, H1, W1] (source image used for prediction, size 32)
    img_smp: [B, 1, H2, W2] (image for sampling, size 44)
    flow: [B, 2, H1, W1] flow predicted from source image pair
    """
    self.img_size = img_size
    H, W = img_size, img_size
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W)
    yy = yy.view(1, H, W)
    self.grid = torch.cat((xx, yy), 0).float()  # [2, H, W]

  def forward(self, flow, img):
    grid = self.grid.repeat(flow.shape[0], 1, 1, 1)  # [bs, 2, H, W]
    if img.is_cuda:
      grid = grid.cuda()
    vgrid = Variable(grid, requires_grad=False) + flow

    vgrid = 2.0 * vgrid / (self.img_size - 1) - 1.0  # max(W-1,1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(img, vgrid)

    return output

class CNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.enc_x = nn.Sequential(
      nn.Conv2d(2, 64, 3, 1, 1, bias=False), # 64 x 28 x 28
      nn.BatchNorm2d(64),
      nn.ELU(),
      nn.AvgPool2d(2, 2, 0), # 64 x 14 x 14

      nn.Conv2d(64, 64, 3, 1, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ELU(),
      nn.Conv2d(64, 64, 3, 1, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ELU(),
      nn.Conv2d(64, 64, 3, 1, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ELU(),
      nn.AvgPool2d(2, 2, 0),  # 64 x 7 x 7
      nn.Conv2d(64, 2, 3, 2, 1), # 2 x 7 x 7
      nn.Tanh()
    )

  def forward(self, x):
    x = self.enc_x(x)
    return x

class DIRNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.vCNN = CNN()
    self.config = config
    self.loss = torch.nn.L1Loss()
    self.loss2 = torch.nn.MSELoss()

    self.upsample4 = nn.Upsample(scale_factor=4, mode='bicubic')
    self.warper2d = Warper2d(config.imh)
    self.LCC = LCC()
    self.Bend_Penalty = Bend_Penalty()
    self.Grad = Grad()
    self.RobustLOSS = RobustLOSS(0.5)

  def forward(self, x, y):
    xy = torch.cat((x, y), dim = 1)
    v = self.vCNN(xy)
    z = WarpST(x, v, self.config.im_size)
    z = z.permute(0, 3, 1, 2)

    robostLoss = self.RobustLOSS.forward(z, y, 0.3)
    loss = robostLoss #self.loss2(z, y) #+ gdloss + bploss # + lccloss #

    return z, v, loss

  def deploy(self, dir_path, x, y):
    with torch.no_grad():
      z, v, _ = self.forward(x, y)
      for i in range(z.shape[0]):
        save_image_with_scale(dir_path+"/{:02d}_x.png".format(i+1), x.cpu().permute(0, 2, 3, 1)[i,:,:,0].numpy())
        save_image_with_scale(dir_path+"/{:02d}_y.png".format(i+1), y.cpu().permute(0, 2, 3, 1)[i,:,:,0].numpy())
        save_image_with_scale(dir_path+"/{:02d}_z.png".format(i+1), z.cpu().permute(0, 2, 3, 1)[i,:,:,0].numpy())

  def deploy_dice(self, x, y, segx, segy):
    with torch.no_grad():
      z, v, _ = self.forward(x, y)
      segz = WarpST(segx, v, self.config.im_size)
      segz = segz.permute(0, 3, 1, 2)
      for i in range(z.shape[0]):
        yield [x.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(), y.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(),
               z.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(),
               segx.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(), segy.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(),
               segz.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy()]

  def deploy_dice_for_plots(self, x, y, segx, segy):
    with torch.no_grad():
      z, v, _ = self.forward(x, y)
      segz = WarpST(segx, v, self.config.im_size)
      segz = segz.permute(0, 3, 1, 2)
      for i in range(z.shape[0]):
        yield [x.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(), y.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(),
               z.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(),
               segx.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(), segy.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(),
               segz.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(),
               v.cpu().permute(0, 2, 3, 1)[i, :, :, :].numpy()
               ]