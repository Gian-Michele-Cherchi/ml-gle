import yaml
import os
import torch

def subtraj_mode_dataset(
                    sourcepath: str, 
                    filepath: str,
                    n_input:int,
                    split: list,
                    nmode:int,
                    norm: bool=True, 
                    device: str="cpu", 
                    test: bool=False
                    ) -> list:
    
    ex_flag = os.path.isfile(filepath)
    assert nmode>= 0, ValueError("mode index cannot be negative!")
    if ex_flag:
        df = torch.load(filepath, map_location=device)
        norms = df["norms"]
        input_train = df["input_train"]
        input_val = df["input_val"]
        target_train = df["target_train"]
        target_val =  df["target_val"]
    else:   
        data = torch.load(sourcepath, map_location=device)
        data = data.swapaxes(0,1)[:,4000:14000,:,:]
        norms = [torch.sqrt((data[:,:n_input]**2).mean(dim=(0,1,3))),torch.sqrt((data[:,:n_input].diff(dim=1)**2).mean(dim=(0,1,3)))]
        nsamples = data.size(1)
        n_train = int(split[0]*nsamples)
        n_val = int(split[1]*nsamples)
        data_pos = data[:,:,nmode:nmode+1,:] #original dataset 
        data_vel = data.diff(dim=1)[:,:,nmode:nmode+1,:] #first d
        input_train = data_pos[:,1:][:,:n_train]
        input_val = data_pos[:,1:][:, n_train:n_val]
        target_train = data_vel[:,:n_train]
        target_val = data_vel[:,n_train:n_val]
        input_test = data_pos[:,1:][:,n_val:]
        target_test = data_vel[:,n_val:]
        df = {"norms": norms, 
            "input_train": input_train, 
            "input_val": input_val, 
            "target_train":target_train, 
            "target_val": target_val,
            "input_test": input_test,
            "target_test": target_test
                }
        torch.save(df, filepath)
        del data 
    
    if norm:
        input_train = input_train/norms[0][nmode]
        input_val = input_val/norms[0][nmode]
        target_train = target_train/norms[1][nmode]
        target_val = target_val/norms[1][nmode]
        
    data_train = torch.cat([input_train[:,:n_input], target_train[:,n_input:n_input+1]], dim=1).squeeze(2)
    data_val = torch.cat([input_val[:,:n_input], target_val[:,n_input:n_input+1]], dim=1).squeeze(2)
    if test:
        data_test = torch.cat([input_test[:,:n_input], target_test[:,n_input:n_input+1]], dim=1).squeeze(2)
        
    for t in range(1,input_train.size(1)-n_input-1):
        tmp = torch.cat([input_train[:,t:t+n_input], target_train[:,t+n_input:t+n_input+1]], dim=1).squeeze(2)
        data_train = torch.cat([data_train, tmp], dim=0)
    for t in range(1,input_val.size(1)-n_input-1):
        tmp = torch.cat([input_val[:,t:t+n_input], target_val[:,t+n_input:t+n_input+1]], dim=1).squeeze(2)
        data_val = torch.cat([data_val, tmp], dim=0)
        
    if test:
        for t in range(1,input_test.size(1)-n_input-1):
            tmp = torch.cat([input_test[:,t:t+n_input], target_test[:,t+n_input:t+n_input+1]], dim=1).squeeze(2)
            data_test = torch.cat([data_test, tmp], dim=0)
            return data_train, data_val, data_test, norms
        
    else:
        return data_train, data_val, norms
        
    #data_train = data_train[torch.randperm(data_train.size(0))]
    #data_val = data_val[torch.randperm(data_val.size(0))]
   

                
