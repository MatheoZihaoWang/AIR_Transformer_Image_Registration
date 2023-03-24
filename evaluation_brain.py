import torch
from data.loader import imageHandle
from models.ops import mkdir

from data.loader import Preprocessing

import torch
import numpy as np
from data.loader import imageHandle

from torch.optim.lr_scheduler import StepLR
from config import get_config
from data.cochleaDataset import cochlea_test_dataset

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

#Model Setup
model = DIRNet(config).to(device)
optim = torch.optim.Adam(model.parameters(), lr = config.lr)
scheduler = StepLR(optim, step_size=200, gamma=0.5)

loader = cochlea_test_dataset()
model.load_state_dict(torch.load(config.ckpt_dir + '/model_weights_trans.pth'))


#Data Setup
processor = Preprocessing()
processor.alginment()
patches_fix_train=[]
patches_fix_test=[]
patches_mov_train=[]
patches_mov_test=[]
stacksize = 4
stack_id_count=0
for stack_id in range(1,stacksize,2):
    try:
        path_fix = './data/brain_registration/OASIS_OAS1_000' + str(
            stack_id) + '_MR1/slice_norm.nii.gz'
        path_mov = './data/brain_registration/OASIS_OAS1_000' + str(
            stack_id+1) + '_MR1/slice_norm.nii.gz'
        processor = Preprocessing(datasets_fix_mask_path=path_fix, datasets_mov_image_path=path_mov)
        processor.alginment()

        mov_train, fix_train = processor.patchlization_for_training(config.imw, False)
        mov_test, fix_test = processor.patchlization_for_testing(config.imw, False)

        mov_train = processor.normalization(mov_train)
        fix_train = processor.normalization(fix_train)
        fix_test = processor.normalization(fix_test)
        mov_test = processor.normalization(mov_test)

        patches_fix_train.append(fix_train[:,np.newaxis,:,:])
        patches_fix_test.append(fix_test[:,np.newaxis,:,:])
        patches_mov_train.append(mov_train[:,np.newaxis,:,:])
        patches_mov_test.append(mov_test[:,np.newaxis,:,:])
        stack_id_count+=1
    except Exception as e:
        print(e)
tensor_fix_train = torch.tensor(patches_fix_train)
tensor_mov_train = torch.tensor(patches_mov_train)
tensor_fix_test = torch.tensor(patches_fix_test)
tensor_mov_test = torch.tensor(patches_mov_test)

train_pr = imageHandle(tensor_mov_train, tensor_fix_train)
test_pr = imageHandle(tensor_fix_test, tensor_mov_test)


batch_xx, batch_yy = test_pr.sample_pair(stack_id_count-1, config.batch_size)
batch_x, batch_y = batch_xx.to(device), batch_yy.to(device)
for res in  model.deploy_softdice(batch_x, batch_y):
    xx = res[0]
    yy = res[1]
    zz = res[2]

    dice1 = np.sum(zz[yy == 1]) * 2.0 / (np.sum(zz) + np.sum(yy))
    dice2 = np.sum(zzSeg[yySeg == 1]) * 2.0 / (np.sum(xxSeg) + np.sum(yySeg))
    DICE_SCORE_BEFORE_TRAN.append(dice2)
    DICE_SCORE_AFTER_TRAN.append(dice1)

    plt.subplot(221);
    plt.imshow(xx);
    plt.subplot(222);
    plt.imshow(yy)
    plt.subplot(223);
    plt.imshow(zz);
    plt.subplot(224);
    plt.imshow(yy);
    plt.show()

    plt.subplot(223);
    plt.imshow(xxSeg);
    plt.subplot(224);
    plt.imshow(yySeg)
    plt.subplot(221);
    plt.imshow(zzSeg);
    plt.subplot(222);
    plt.imshow(yySeg);
    plt.show()