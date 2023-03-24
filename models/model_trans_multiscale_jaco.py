from einops.layers.torch import Rearrange
from models.ops import *
from models.WarpST import WarpST
from models.STN import STNet
from models.torch_similarity.functional import normalized_cross_correlation
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import constrains as cons
import torch.nn.functional as F
from config import get_config

def plot(im):
  im = np.array(im.tolist())
  plt.imshow(im, cmap='gray', vmin=0, vmax=1)
  plt.show()
  return None

def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.001

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, out_dim, dim, depth, heads, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_sz = patch_size
        self.img_sz = image_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        #self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        #self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            #nn.LayerNorm(dim),
            nn.Linear(in_features=dim,  out_features=out_dim),
            nn.Tanh()
        )
        self.conv = nn.Sequential(nn.Conv2d(2, 2, 2, 1, 1))
        self.sin_embedding = PositionalEncoding(d_model=dim)

    def forward(self, img_fix, img_mov):

        x_fix = self.to_patch_embedding(img_fix)
        b, n, pp = x_fix.shape

        #x_fix = self.sin_embedding(x_fix)
        x_fix += self.pos_embedding[:, :(n + 1)]
        #x_fix = self.dropout(x_fix)

        x_mov = self.to_patch_embedding(img_mov)
        b, n, pp = x_mov.shape #Size batch, size channels, patch_size^2

        #x_mov = self.sin_embedding(x_mov)
        x_mov += self.pos_embedding[:, :(n + 1)]
        #x_mov = self.dropout(x_mov)

        memory = self.transformer_encoder(x_fix)
        vf1 = self.transformer_decoder(x_mov, memory) # Get the feature map pf velocity field
        vf = self.to_latent(vf1)
        vf = self.mlp_head(vf)
        vf = torch.reshape(torch.swapaxes(vf, -2, -1), (b, 2, np.int(self.img_sz/self.patch_sz), np.int(self.img_sz/self.patch_sz)))  # _,_,image_size/patch_size
        return vf


class DIRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ViT_1 = ViT(image_size = config.imw, patch_size = 8, out_dim = 2, dim = 128, depth = 1, heads = 128, dropout = 0.0, emb_dropout = 0.1)
        self.ViT_4 = ViT(image_size = config.imw, patch_size = 16, out_dim = 2, dim = 128, depth = 1, heads = 64, dropout = 0.0, emb_dropout = 0.1)
        self.ViT_8 = ViT(image_size = config.imw, patch_size = 32, out_dim = 2, dim = 128, depth = 1, heads = 32, dropout = 0.0, emb_dropout = 0.1)

        self.config = config
        self.loss2 = torch.nn.MSELoss()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bicubic')
        self.register_parameter(name='w1', param=torch.nn.Parameter(torch.tensor([0.1])))
        self.register_parameter(name='w2', param=torch.nn.Parameter(torch.tensor([0.6])))
        self.register_parameter(name='w3', param=torch.nn.Parameter(torch.tensor([0.3])))
# Dim = 4 bu shou lian; Dim  = 16 shou lian
    def forward(self, x, y):
        v1 = self.ViT_1(x, y)
        v4 = self.ViT_4(x, y)
        v8 = self.ViT_8(x, y)
        v4 = self.upsample2(v4)
        v8 = self.upsample4(v8)

        v = (self.w1 * v1 + self.w2 * v4 + (1- self.w1 - self.w2) * v8)
        v_jaco = []
        v = v.permute(0, 2, 3, 1)
        for idlen in range(v.shape[0]):
            jac = cons.compute_Jaco(v[idlen, :].cpu().detach().numpy())
            v_jaco.append(jac)
        v_jaco = torch.tensor(v_jaco) #nb,w,h
        v = v.permute(0, 3, 1, 2)

        z = WarpST(x, v, self.config.im_size)
        z = z.permute(0, 3, 1, 2)

        loss = self.loss2(z, y) + torch.mean(F.relu(-1 * v_jaco))
        #loss = -normalized_cross_correlation(z, y, False)
        return z, loss

    def deploy(self, dir_path, x, y):
        with torch.no_grad():
          z, _ = self.forward(x, y)
          for i in range(z.shape[0]):
            save_image_with_scale(dir_path+"/{:02d}_x.png".format(i+1), x.cpu().permute(0, 2, 3, 1)[i,:,:,0].numpy())
            save_image_with_scale(dir_path+"/{:02d}_y.png".format(i+1), y.cpu().permute(0, 2, 3, 1)[i,:,:,0].numpy())
            save_image_with_scale(dir_path+"/{:02d}_z.png".format(i+1), z.cpu().permute(0, 2, 3, 1)[i,:,:,0].numpy())

    def deploy_dice(self, x, y):
        with torch.no_grad():
            z, _ = self.forward(x, y)
            for i in range(z.shape[0]):
                yield [x.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(), y.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy(), z.cpu().permute(0, 2, 3, 1)[i, :, :, 0].numpy()]
