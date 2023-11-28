#########################################CONFIG 
import os
import yaml
import torch 
import logging
logging.basicConfig(level=logging.INFO)
from neural_ar.model import *
from utils import *
from data.dataset import *
import  tqdm
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open("config/train/train_conf.yaml") as f:
    train_conf = yaml.load(f, Loader=yaml.FullLoader)
temp = train_conf["temp"]
n_train = train_conf["per_train"]
k = train_conf["k"]

base_T = 300
res_T = 10 
NTEMPS = 11
NCHAINS = 100 
NBEADS = 100
#NMODES = 100 
PROJECTPATH = config["paths"]["PROJECTPATH"]
DATAPATH = config["paths"]["DATAPATH"]
SAVEPATH = config["paths"]["SAVEPATH"]
DEVICE = config["paths"]["DEVICE"]

modes_corr = {}
open_log = tqdm.tqdm(total=0, position=0, bar_format='{desc}')
open_log.set_description_str(
    f'[Simulation Boxes Analysis: '+str(base_T)+"K - "+str(base_T + int(res_T*(NTEMPS-1)))+"K]"
    )
boxes_bar = tqdm.tqdm(total=NTEMPS, desc="Sim Boxes", position=1)
for i in range(1,NTEMPS+1):
    # data load 
    temp = base_T + int(res_T*(i-1))
    sourceparth = os.path.join(DATAPATH, "modesdata_T"+str(temp)+"_.pt")
    modes = torch.load(sourceparth, map_location=DEVICE).swapaxes(0,1)
    
    #####################################################################################AUTOCORR
    autocorr_bar = tqdm.tqdm(total=k, desc="Autocorr, T="+str(temp)+"K", position=3)
    autocorr_box = torch.zeros(k, modes.size(1)).to(modes)
    for nmod in range(1,k):
        temp = {}
        acf = autocorrFFT(modes[:,:n_train,nmod].swapaxes(1,2).contiguous().view(modes.size(0)*3,-1))/modes[:,:n_train,nmod].var()
        temp["nacfs"] = acf.mean(dim=0)
        temp["SE"] = torch.tensor(list(map(lambda k: acf_se(acf[:,:k],acf.size(1)), range(1,acf.size(1)))))
   
        modes_corr["mode"+str(nmod)] = temp
        autocorr_bar.update(1)
        
    box_acorr_savepath = os.path.join(DATAPATH, "md_baselines")
    torch.save(modes_corr, os.path.join(box_acorr_savepath,"modes_nacfs_T"+str(temp)+"_ntrain"+str(n_train)+".pt"))
    del autocorr_box
  
    boxes_bar.update(1)
    del modes
    





    
    