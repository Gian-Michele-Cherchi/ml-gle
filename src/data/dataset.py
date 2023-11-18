import yaml
import os
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
PROJECTNAME = config["PROJECTNAME"]
USERPATH = config["USERPATH"]
SYSTEM = config["SYSTEM"]
FULLPATH  = os.path.join(USERPATH, PROJECTNAME)
import sys 
sys.path.append(FULLPATH)
#######################################
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import rouse_matrix3D, rouse_matrix1D
from src.utils import makedirs
import numpy as np 
import gc 
from torch.distributions import Normal
############################################# CONFIG
TRAIN_SPLIT_PERCENTAGE = 0.7 
VAL_SPLIT_PERCENTAGE = 0.8
USERPATH = "/home/gmcherch/project"
######################################################


def get_modes_data(data,input_seq,train_val,nmode,norms: bool=True, not_seq:bool=True , gen_x: bool=False):
    if norms:
        norms = [torch.sqrt((data**2).mean(dim=(0,1,3))),torch.sqrt((data.diff(dim=1)**2).mean(dim=(0,1,3)))]
        #dataset = dataset[torch.randperm(dataset.size(0))]
        data_pos = data[:,:,nmode:nmode+1,:]/norms[0][nmode] #original dataset 
        if not gen_x:
            data_vel = data.diff(dim=1)[:,:,nmode:nmode+1,:]/norms[1][nmode] #first d
            nsamples = data.size(1)
            n_train = int(train_val[0]*nsamples)
            n_val = int(train_val[1]*nsamples)
            input_train = data_pos[:,1:][:,:n_train]
            input_val = data_pos[:,1:][:, n_train:n_val]
            target_train = data_vel[:,:n_train]
            target_val = data_vel[:,n_train:n_val]
            input_test = data_pos[:,1:][:,n_val:]
            target_test = data_vel[:,n_val:]
        else:
            data_vel = data[:,:,nmode:nmode+1,:]/norms[0][nmode]
            nsamples = data.size(1)
            n_train = int(train_val[0]*nsamples)
            n_val = int(train_val[1]*nsamples)
            input_train = data_pos[:,:][:,:n_train]
            input_val = data_pos[:,:][:, n_train:n_val]
            target_train = data_vel[:,:n_train]
            target_val = data_vel[:,n_train:n_val]
            input_test = data_pos[:,:][:,n_val:]
            target_test = data_vel[:,n_val:]
    else:
        data_pos = data[:,:,nmode:nmode+1,:] #original dataset 
        if not gen_x:
            data_vel = data.diff(dim=1)[:,:,nmode:nmode+1,:] 
        else:
            data_vel = data[:,:,nmode:nmode+1,:]
    if not_seq:
        data_train = torch.cat([input_train[:,:input_seq], target_train[:,input_seq:input_seq+1]], dim=1).squeeze(2)
        data_val = torch.cat([input_val[:,:input_seq], target_val[:,input_seq:input_seq+1]], dim=1).squeeze(2)
        data_test = torch.cat([input_test[:,:input_seq], target_test[:,input_seq:input_seq+1]], dim=1).squeeze(2)
        for t in range(1,input_train.size(1)-input_seq-1):
            tmp = torch.cat([input_train[:,t:t+input_seq], target_train[:,t+input_seq:t+input_seq+1]], dim=1).squeeze(2)
            data_train = torch.cat([data_train, tmp], dim=0)
        for t in range(1,input_val.size(1)-input_seq-1):
            tmp = torch.cat([input_val[:,t:t+input_seq], target_val[:,t+input_seq:t+input_seq+1]], dim=1).squeeze(2)
            data_val = torch.cat([data_val, tmp], dim=0)
        for t in range(1,input_test.size(1)-input_seq-1):
            tmp = torch.cat([input_test[:,t:t+input_seq], target_test[:,t+input_seq:t+input_seq+1]], dim=1).squeeze(2)
            data_test = torch.cat([data_test, tmp], dim=0)
        #data_train = data_train[torch.randperm(data_train.size(0))]
        #data_val = data_val[torch.randperm(data_val.size(0))]
        del input_train, target_train ,input_val, target_val, data_pos, data_vel, input_test, target_test
        return data_train, data_val, data_test, norms
    else:
        input = data_pos[:,1:]
        target = data_vel
        data = torch.cat([torch.cat([input.unsqueeze(1), target.unsqueeze(1)], dim=1)[:,:,:5000], 
                  torch.cat([input.unsqueeze(1), target.unsqueeze(1)], dim=1)[:,:,5000:10000]], dim=0)
        data = data[torch.randperm(data.size(0))]
        del input
        del target 
        del data_pos, data_vel
        ntrain = int(0.8*nsamples)
        nval = int(0.99*nsamples)
        train_data = data[:ntrain]
        val_data = data[ntrain:nval]
        del data
        return train_data, val_data, norms
        

                
