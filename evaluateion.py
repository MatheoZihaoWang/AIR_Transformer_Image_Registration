import torch
import numpy as np
from data.loader import imageHandle
from utils.matric import soft_dice_loss
from torch.optim.lr_scheduler import StepLR
from config import get_config
from data.cochleaDataset import cochlea_test_dataset
from scipy.spatial import distance

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = get_config(is_train=True)
if config.cnn:
    from models.models import DIRNet
    print('USING CNN')
else:
    from models.model_trans_multiscale import DIRNet
    print('USING TRNSFORMER MULTI')

torch.manual_seed(0)
train_batch = 100000
test_batch = 20000
eps = 0.1
#Model Setup
model = DIRNet(config).to(device)
optim = torch.optim.Adam(model.parameters(), lr = config.lr)
scheduler = StepLR(optim, step_size=200, gamma=0.5)

loader = cochlea_test_dataset()
if config.cnn:
    model.load_state_dict(torch.load(config.ckpt_dir + '/model_weights_cnn.pth'))
else:
    model.load_state_dict(torch.load(config.ckpt_dir + '/model_weights_trans.pth'))
DICE_SCORE_BEFORE_TRAN = []
DICE_SCORE_AFTER_TRAN = []

SOFT_SCORE_BEFORE_TRAN = []
SOFT_SCORE_AFTER_TRAN = []

MSE_SCORE_BEFORE_TRAN = []
MSE_SCORE_AFTER_TRAN = []

for (fix, mov, fixseg, movseg) in loader.get_pairs():
    batch_xx, batch_yy = torch.tensor(fix[:, np.newaxis, :, :]), torch.tensor(mov[:, np.newaxis, :, :])
    batch_xxSEG, batch_yySEG = torch.tensor(fixseg[:, np.newaxis, :, :]), torch.tensor(movseg[:, np.newaxis, :, :])
    batch_x, batch_y = batch_xx.to(device), batch_yy.to(device)
    batch_xxSEG, batch_yySEG = batch_xxSEG.to(device), batch_yySEG.to(device)

    for res in model.deploy_dice(batch_y, batch_x, batch_yySEG, batch_xxSEG):
        xx = res[0]
        yy = res[1]
        zz = res[2]
        xxSeg = res[3]
        yySeg = res[4]
        zzSeg = res[5]
        xxSeg[xxSeg > eps] = 1
        yySeg[yySeg > eps] = 1
        zzSeg[zzSeg > eps] = 1

        xxSeg[xxSeg <= eps] = 0
        yySeg[yySeg <= eps] = 0
        zzSeg[zzSeg <= eps] = 0


        dice1 = np.sum(zzSeg[yySeg == 1]) * 2.0 / (np.sum(zzSeg) + np.sum(yySeg))
        dice2 = np.sum(xxSeg[yySeg == 1]) * 2.0 / (np.sum(xxSeg) + np.sum(yySeg))
        DICE_SCORE_BEFORE_TRAN.append(dice2)
        DICE_SCORE_AFTER_TRAN.append(dice1)

        SOFT_SCORE_BEFORE_TRAN.append(scipy.spatial.distance.directed_hausdorff(xxSeg, yySeg)[0] +
                                      scipy.spatial.distance.directed_hausdorff(yySeg, xxSeg)[0])
        SOFT_SCORE_AFTER_TRAN.append(scipy.spatial.distance.directed_hausdorff(zzSeg, yySeg)[0] +
                                     scipy.spatial.distance.directed_hausdorff(yySeg, zzSeg)[0])

        MSE_SCORE_BEFORE_TRAN.append(dice2)
        MSE_SCORE_AFTER_TRAN.append(dice1)

        MSE_SCORE_BEFORE_TRAN.append(mean_squared_error(xx, yy))
        MSE_SCORE_AFTER_TRAN.append(mean_squared_error(xx, zz))

        plt.subplot(221);plt.imshow(xx);plt.subplot(222);plt.imshow(yy)
        plt.subplot(223);plt.imshow(zz);plt.subplot(224);plt.imshow(yy);plt.show()

        plt.subplot(221);plt.imshow(xxSeg);plt.subplot(222);plt.imshow(yySeg)
        plt.subplot(223);plt.imshow(zzSeg);plt.subplot(224);plt.imshow(yySeg);plt.show()

print('Before Dice:' + str(np.mean(DICE_SCORE_BEFORE_TRAN)))
print('After Dice:' + str(np.mean(DICE_SCORE_AFTER_TRAN)))
print('Before hausdorff:' + str(np.mean(SOFT_SCORE_BEFORE_TRAN)))
print('After hausdorff:' + str(np.mean(SOFT_SCORE_AFTER_TRAN)))
print('Before mse:' + str(np.mean(MSE_SCORE_BEFORE_TRAN)))
print('After mse:' + str(np.mean(MSE_SCORE_AFTER_TRAN)))


print('Before Dice:' + str(np.std(DICE_SCORE_BEFORE_TRAN)))
print('After Dice:' + str(np.std(DICE_SCORE_AFTER_TRAN)))
print('Before hausdorff:' + str(np.std(SOFT_SCORE_BEFORE_TRAN)))
print('After hausdorff:' + str(np.std(SOFT_SCORE_AFTER_TRAN)))
print('Before mse:' + str(np.std(MSE_SCORE_BEFORE_TRAN)))
print('After mse:' + str(np.std(MSE_SCORE_AFTER_TRAN)))