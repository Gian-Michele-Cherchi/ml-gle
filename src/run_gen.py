##########################################CONFIG & LIBS
import os
import yaml
import sys 
import gc
import torch 
from ia.library.dynamical_network import  DyamicalScoreSDE_3D
from ia.library.utils import *
from ia.dataset import *
from ia.library.training import *
from ia.library.sampling import *
import logging
import functools
from src.utils import *
logging.basicConfig(level=logging.INFO)
# changing device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
PROJECTNAME = config["PROJECTNAME"]
USERPATH = config["USERPATH"]
SYSTEM = config["SYSTEM"]
FULLPATH  = os.path.join(USERPATH, PROJECTNAME)
SUBMODULE = os.path.join(FULLPATH, "ia")
sys.path.append(SUBMODULE)
DATAPATH = os.path.join(FULLPATH,"data")
#############################################
seed = 12345
torch.manual_seed(seed)
############################MODEL 
ACTIVATION = 'relu'
INPUT_SEQ = 128
NSAMPLE_STEPS = 1e4
NMAX_STEPS = int(2e3)
IN_CHANNELS = 1
# SDE Setup 
sigma = 10.
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

# Dynamical Model 
scoreonly = False
gaussonly= False
reg = 0
split = 0.1
model = DyamicalScoreSDE_3D( 
                    input_length=INPUT_SEQ, 
                    in_channels=1, 
                    residual_channels=4,
                    marginal_prob_std_fn=marginal_prob_std_fn,
                    pred_var_dims=3,
                    scoreonly=scoreonly,
                    score_nlayers=4,
                    reg=reg
                    ).to(device)

#N_MODES = 1
N_TRAIN = 1000
NGENSTEP = 100000
SAVEPATH = os.path.join(FULLPATH,"save/melt/temp/T300/")
dataset = torch.load(os.path.join(FULLPATH, "data/melt/temp/ready/modesdata_T300_.pt"), map_location=device).swapaxes(0,1)[:,5000:15000,:,:]
norms = [torch.sqrt((dataset[:,:N_TRAIN]**2).mean(dim=(0,1,3))),
         torch.sqrt((dataset[:,:N_TRAIN].diff(dim=1)**2).mean(dim=(0,1,3)))]


mode_corr = []
genmode_corr = []
for nmode in range(1,2):
    LOSSPATH = os.path.join(SAVEPATH, "losses_"+str(nmode)+"modes_alpha"+str(reg)+"_split"+str(split)+"_.csv")
    losses = pd.read_csv(LOSSPATH)
    #epoch = np.argmin(losses["Val Dyn"].values)+1
    if nmode == 1:
        epoch = 24
    print(epoch)
    CHECKPTFILE = "checkpt_"+str(nmode)+"mode_alpha"+str(reg)+"_"+str(epoch)+"_split"+str(split)+"_.pth"
    BESTCKPTFILE = "checkpt_best_"+str(nmode)+"mode_alpha"+str(reg)+"_split"+str(split)+"_.pth"
    CHECKPTPATH = os.path.join(SAVEPATH,CHECKPTFILE)
    checkpoint = torch.load(CHECKPTPATH)
    model.load_state_dict(checkpoint)
    model.eval()
    norms_modes = (norms[1][nmode]/norms[0][nmode])
    nsamples = 300
    dataset_pos = dataset[:,:,nmode]/norms[0][nmode]
    # N polymers x data source x n steps x r rouse modes x 3 
    _, _, test_data ,norms = get_modes_data(data=dataset, 
                                            input_seq=INPUT_SEQ,
                                            train_val=[split, split +0.625*split], 
                                            nmode=nmode, 
                                            norms=True, 
                                            not_seq=True,
                                            gen_x=False
                                            )
    gen_data, mu_gen, sigma_gen     = DynamicsIntegration_new(
                                        model, 
                                        norms=norms_modes,
                                        in_trajectories=test_data[:nsamples,:INPUT_SEQ],
                                        NGENSTEP=NGENSTEP,
                                        device=device,
                                        IN_CHANNELS=1,
                                        sampling=None,
                                        std=norms[0][nmode]
                                        )
    #mode_corr.append(autocorrFFT(dataset_pos[:,:INPUT_SEQ+NGENSTEP].swapaxes(1,2).contiguous().view(dataset.size(0)*3,-1)).mean(dim=0)/dataset_pos[:,:INPUT_SEQ+NGENSTEP].var())
    #genmode_corr.append(autocorrFFT(gen_data[:,:,0].swapaxes(1,2).contiguous().view(3*gen_data.size(0),-1)).mean(dim=0)/gen_data[:,:,0].var())
    torch.save(gen_data, os.path.join(SAVEPATH, "gen_data/long/gen_data_mode"+str(nmode)+"_alpha"+str(reg)+"_epoch"+str(epoch)+"_split"+str(split)+"_.pt"))