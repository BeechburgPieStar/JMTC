import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.train_tools import *
import torch.nn.functional as F
import numpy as np
import random

# hyperparameters
batch_size  = 128
epochs  = 30
lr  = 0.001
wd = 0
seed = 2023
train_ratio = 0.3
loss_lambda = 0.5
env = "AWGN"#Dynamic
modelname  = "CNN"

setup_seed(seed)
if env == "AWGN":
    model = get_model(modelname, 2, 9, 11).cuda()
elif env == "Dynamic":
    model = get_model(modelname, 2, 6, 8).cuda()

model.cuda()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = wd)
loss = [nn.NLLLoss().cuda(),nn.NLLLoss().cuda()]
x_train, x_val, y1_train, y1_val, y2_train, y2_val = gen_train_dataset(env, train_ratio)
train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y1_train), torch.Tensor(y2_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y1_val), torch.Tensor(y2_val))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


loss=train_and_evaluate(model, 
        loss_function=loss, 
        loss_lambda = loss_lambda,
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        optimizer=optim, 
        epochs=epochs, 
        save_path=f'weights/{modelname}_{env}_{loss_lambda}.pth')
from scipy.io import savemat
savemat(f'losses/{modelname}_{env}_{loss_lambda}.mat',{"loss":loss})

for snr in range(-20, 20, 2):
    x_test, y1_test, y2_test = gen_test_dataset_per_snr(env, snr) 
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y1_test), torch.Tensor(y2_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = torch.load(f'weights/{modelname}_{env}_{loss_lambda}.pth')
    test(model,test_dataloader)
