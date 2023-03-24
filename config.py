class Config(object):
  pass

def get_config(is_train):
  config = Config()
  config.imh = config.imw = 64
  if is_train:
    config.FASION = False
    config.batch_size = 1#128
    config.im_size = [config.imh, config.imw]
    config.lr = 0.1e-3 #1.0e-2; 1.0e-2 for CNN
    config.iteration = 10000
    config.tmp_dir = "./temp/tmp"
    config.ckpt_dir = "./temp/ckpt"
    config.cnn = False
    config.nThreads = 2
  else:
    config.batch_size = 2
    config.im_size = [config.imh , config.imw]
    config.result_dir = "./DIRNet/result"
    config.ckpt_dir = "./DIRNet/ckpt"
  return config


