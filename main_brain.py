import torch
from data.loader import imageHandle
from models.ops import mkdir
from torch.optim.lr_scheduler import StepLR
from config import get_config
from data.loader import Preprocessing
import numpy as np
#Prepare parameters and environment
import matplotlib.pyplot as plt
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

mkdir(config.tmp_dir)
mkdir(config.ckpt_dir)


#Model Setup
model = DIRNet(config).to(device)
optim = torch.optim.Adam(model.parameters(), lr = config.lr)
scheduler = StepLR(optim, step_size=200, gamma=0.5)

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

#Ready, GO!
total_loss = 0
for i in range(config.iteration):
    batch_xx, batch_yy = train_pr.sample_pair(stack_id_count-1, config.batch_size)
    batch_x, batch_y = batch_xx.to(device), batch_yy.to(device)
    optim.zero_grad()
    _, _, loss = model(batch_x, batch_y)
    loss.backward()
    optim.step()
    scheduler.step()
    total_loss += loss

    if (i+1) % 100 == 0:
      print("iter {:>6d} : {}".format(i + 1, total_loss))
      total_loss = 0
      batch_xx, batch_yy = test_pr.sample_pair(stack_id_count-1, config.batch_size)
      batch_x, batch_y = batch_xx.to(device), batch_yy.to(device)
      model.deploy(config.tmp_dir, batch_x, batch_y)
      if not config.cnn:
        torch.save(model.state_dict(), config.ckpt_dir + '/model_weights_trans.pth')
      else:
        torch.save(model.state_dict(), config.ckpt_dir + '/model_weights_cnn.pth')


plt.imshow(batch_yy[2,0,:,:]);plt.show()
plt.imshow(batch_xx[2,0,:,:]);plt.show()

baz = model.forward(batch_x, batch_y)