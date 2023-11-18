import os 
import numpy as np
import xarray as xr
import torch
from scipy.fftpack import dct
import logging
import time 
import yaml
import sys 
import gc
import tqdm
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PROJECTNAME = config["PROJECTNAME"]
USERPATH = config["USERPATH"]
SYSTEM = config["SYSTEM"]
FULLPATH  = os.path.join(USERPATH, PROJECTNAME)
SUBMODULE = os.path.join(FULLPATH, "ia")
sys.path.append(SUBMODULE)
logging.basicConfig(level=logging.INFO)

###################################
def xarray2torch(sourcepath: str, 
                 destpath: str,
                 device: str="cuda",
                 save_modes: bool=True,
                 nsteps: int=1000,
                 **kwargs
                 ):
    NBEADS = kwargs["nbeads"]
    NCHAINS = kwargs["nchains"]
    temp = kwargs["T"]
    NMODES = NBEADS 
    d =  xr.open_dataset(sourcepath)
    pos = d.unwrapped_coordinates.drop_vars(("cell_spatial", "spatial","cell_angular"))
    pos = pos.coarsen(atom=NBEADS).construct(atom=('polymer','grain'))
    
    modes_dataset = torch.zeros(nsteps, NCHAINS, NBEADS, 3, device=device)
    posdata =  torch.zeros(nsteps, NCHAINS, NBEADS, 3, device=device)
    
    pol_bar = tqdm.tqdm(total=100, desc="Polymers", position=2)
    for npol in range(NCHAINS):
        if save_modes:
            #print("#", npol+1, '/', npol, "({:1f} s)".format(time.time() - tstart))
            mode = dct(pos.isel(polymer=npol).values, type=2, axis=1)[:nsteps,:NMODES]/(2*NBEADS)
            modes_dataset[:,npol,:,:] = torch.tensor(mode).to(modes_dataset)
        else:
            #print("#", npol+1, '/', npol, "({:1f} s)".format(time.time() - tstart))
            polymerdata = dct(pos.isel(polymer=npol).values, type=2, axis=1)[:nsteps,:NMODES]
            posdata[:,npol,:,:] = torch.tensor(polymerdata).to(posdata)
        pol_bar.update(1)
        
    if save_modes:
        filename = "modesdata_T"+str(temp)+"_.pt"
        torch.save(modes_dataset,os.path.join(destpath, filename))
        del modes_dataset
    else:
        filename = "posdata_T"+str(temp)+"_.pt"
        torch.save(posdata,os.path.join(destpath, filename))
        del posdata
    torch.cuda.empty_cache()

tstart = time.time()

ntemps = 11
base_T = 300
boxes_bar = tqdm.tqdm(total=ntemps, desc="Sim Boxes", position=0)
source_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
for i in range(1,ntemps+1):       
    # data conversion function call
    if i <10: num = "0"+str(i) 
    else: num = str(i)
    sourcepath = os.path.join(FULLPATH, "data/melt/temp/raw/simul0"+num+"/dump.nc")
    destpath   = os.path.join(FULLPATH, "data/meltBR/temp/ready")
    source_log.set_description_str(f'Current Source: {sourcepath}')
    xarray2torch(sourcepath=sourcepath,
                 destpath=destpath,
                 nbeads=100,
                 nchains=100,
                 save_modes=False,
                 nsteps=1000, 
                 T=base_T + int(10*(i-1)),
                 )
    boxes_bar.update(1)
print("Elapsed Time: {:1f} s".format(time.time() - tstart))
    
logging.info("[Done]")