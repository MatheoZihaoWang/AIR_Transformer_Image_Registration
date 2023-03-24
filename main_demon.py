import torch
import numpy as np
from models.ops import mkdir
from torch.optim.lr_scheduler import StepLR
from config import get_config
from data.cochleaDataset import cochlea_dataset
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
if config.cnn:
    print("CNN have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
else:
    print("Transformer have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

optim = torch.optim.Adam(model.parameters(), lr = config.lr)
scheduler = StepLR(optim, step_size=200, gamma=0.5)

loader = cochlea_dataset()

total_loss = 0

for i in range(config.iteration):
    if i < 300:
        batch_xx, batch_yy = loader.get_pairs(warmup=True)
    else:
        batch_xx, batch_yy = loader.get_pairs(warmup=False)
    batch_xx, batch_yy = torch.tensor(batch_xx[:,np.newaxis,:,:]), torch.tensor(batch_yy[:,np.newaxis,:,:])
    batch_x, batch_y = batch_xx.to(device), batch_yy.to(device)
    optim.zero_grad()
    _, v, loss = model(batch_x, batch_y)
    loss.backward()
    optim.step()
    scheduler.step()
    total_loss += loss

    if (i+1) % 100 == 0:
      print("iter {:>6d} : {}".format(i + 1, total_loss))
      total_loss = 0
      batch_xx, batch_yy = loader.get_pairs(warmup=False)
      batch_xx, batch_yy = torch.tensor(batch_xx[:,np.newaxis,:,:]), torch.tensor(batch_yy[:,np.newaxis,:,:])
      batch_x, batch_y = batch_xx.to(device), batch_yy.to(device)
      model.deploy(config.tmp_dir, batch_x, batch_y)
      if not config.cnn:
        torch.save(model.state_dict(), config.ckpt_dir + '/model_weights_trans.pth')
      else:
        torch.save(model.state_dict(), config.ckpt_dir + '/model_weights_cnn.pth')

plt.imshow(batch_yy[2,0,:,:]);plt.show()
plt.imshow(batch_xx[2,0,:,:]);plt.show()
