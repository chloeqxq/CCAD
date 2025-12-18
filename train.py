from share import *
import torch
import random
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mvtecad_dataloader import MVTecDataset
from sgn.logger import ImageLogger
from sgn.model import create_model, load_state_dict
from visa_dataloader import VisaDataset
from mvtecad_loco_dataloader import MVTecLocoDataset
from mvtecad_3d_dataloader import MVTec3dDataset
from mtd_dataloader import MTDDataset
from dagm_dataloader import DAGMDataset
from pytorch_lightning.callbacks import ModelCheckpoint

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministirc = True
    torch.backends.cudnn.benchmark = False

# Configs
resume_path = './models/diad.ckpt'

setup_seed(1)
batch_size = 12
logger_freq = 3000000000000
learning_rate = 1e-4
only_mid_control = True
# data_path = '/root/autodl-tmp/mvtecad/'
data_path = '/apps/data2/adminad/AD_data/MVTec/'
# data_path = '/apps/data2/adminad/AD_data/VisA/'
# data_path = '/apps/data2/adminad/AD_data/MVTec_loco/'
# data_path = '/apps/data2/adminad/AD_data/MVTec_3d/'
# data_path = '/apps/data2/adminad/AD_data/MTD_Train_Test_Split/'
# data_path = '/apps/data2/adminad/AD_data/DAGM2007_mini/'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/diad.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# Misc
train_dataset, test_dataset = MVTecDataset('train',data_path), MVTecDataset('test',data_path)
# train_dataset, test_dataset = VisaDataset('train',data_path), VisaDataset('test',data_path)
# train_dataset, test_dataset = MVTecLocoDataset('train',data_path), MVTecLocoDataset('test',data_path)
# train_dataset, test_dataset = MVTec3dDataset('train',data_path), MVTec3dDataset('test',data_path)
# train_dataset, test_dataset = MTDDataset('train',data_path), MTDDataset('test',data_path)
# train_dataset, test_dataset = DAGMDataset('train',data_path), DAGMDataset('test',data_path)
train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=1, shuffle=True)

ckpt_callback_val_loss = ModelCheckpoint(monitor='val_acc', dirpath='./val_ckpt/',mode='max')
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=[1], precision=32, callbacks=[logger,ckpt_callback_val_loss], max_epochs = 200, accumulate_grad_batches = 4, check_val_every_n_epoch = 10)

# Train!
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)