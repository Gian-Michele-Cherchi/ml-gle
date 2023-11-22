import os
import yaml
import sys 
with open("../config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PROJECTNAME = config["PROJECTNAME"]
USERPATH = config["USERPATH"]
SYSTEM = config["SYSTEM"]
FULLPATH  = os.path.join(USERPATH, PROJECTNAME)
SUBMODULE = os.path.join(FULLPATH, "ia")
sys.path.append(SUBMODULE)
##########################################LIB
import torch 
from src.neural_ar.model import *
from src.utils import *
from src.data.dataset import *
from src.neural_ar.training import *
import logging
import functools
import argparse
import wandb
logging.basicConfig(level=logging.INFO)
#########################################CONFIG 
parser = argparse.ArgumentParser()
parser.add_argument('--input_seq', type=int, default=256)
parser.add_argument('--modes', type=int, default=1, nargs='+', required=False)
parser.add_argument('--epochs', type=int, default=2, required=False)
parser.add_argument('--train_batch', type=int, default=128)
parser.add_argument('--val_batch', type=int, default=256)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--save', type=bool, required=False, default=False)
parser.add_argument('--lr', type=float, default=1e-4, required=False)
parser.add_argument('--data', type=str, required=False, default="modesdata_T300_.pt")
parser.add_argument('--experiment', type=str, required=False, default="melt/temp/T300")
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--checkpt', type=list, required=False, default=0)
parser.add_argument('--train_splits', type=float, default= 0.05, required=False, nargs='+')
parser.add_argument('--res_layers', type=int, default=4, required=False, nargs='+')
parser.add_argument("--device", type=str, default="cuda:0", required=False)
args = parser.parse_args()
SAVEPATH = os.path.join(FULLPATH, "save")
SAVEPATH =os.path.join(SAVEPATH,args.experiment)
FILEDATA = os.path.join(FULLPATH, "data/melt/temp/ready/"+args.data)
print("Saving checkpoints to {:s}" .format(SAVEPATH))
device = args.device
##################################################MODEL 
ACTIVATION = 'relu'
learning_rate = args.lr
INPUT_SEQ = args.input_seq
modes =args.modes
EPOCHS = args.epochs
train_batch_size = args.train_batch
test_batch_size = args.val_batch
train_splits = args.train_splits
resume = args.resume
gen_x = args.gen_x
regs = args.regs
score_nalyers = args.score_nlayers
save= args.save
scoreonly = args.scoreonly
res_layers = args.res_layers
if not isinstance(regs, list):
    regs = [regs]
if not isinstance(train_splits, list):
    train_splits = [train_splits]
if not isinstance(modes, list):
    modes = [modes]
if isinstance(res_layers, list) and len(res_layers) != 1:
    assert len(res_layers) == len(train_splits)
else:
    res_layers = [res_layers for _ in range(len(train_splits))]
if isinstance(score_nalyers, list) and len(score_nalyers) != 1:
    assert len(score_nalyers) == len(train_splits)
else:
    score_nalyers = [score_nalyers for _ in range(len(train_splits))]
#######################
seed = args.seed
#torch.manual_seed(seed)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="ml-gle",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)
########################################################## POLYMER MELT MODES - NON MARKOVIAN 
#dataset = torch.load(os.path.join(FULLPATH, "data/melt/temp/ready/Rouse20_Data10k.pt"), map_location=device)
#full_data = torch.load(os.path.join(FULLPATH, "data/melt/temp/ready/modesdata_T400_.pt"), map_location=device)
full_data = torch.load(FILEDATA, map_location=device)
dataset = full_data.swapaxes(0,1)[:,4000:14000,:,:]
del full_data
torch.cuda.empty_cache()
for split, layers, score_layers in zip(train_splits, res_layers, score_nalyers):
    for mode in modes:
        
        train_data, val_data, _ ,norms = get_modes_data(data=dataset, 
                                                input_seq=INPUT_SEQ,
                                                train_val=[split, split +0.625*split], 
                                                nmode=mode, 
                                                norms=True, 
                                                not_seq=True,
                                                gen_x=gen_x
                                                    )
        
        train_dl = DataLoader(train_data, train_batch_size, shuffle=True)
        val_dl = DataLoader(val_data, test_batch_size, shuffle=True)
        train_loader = DeviceDataLoader(train_dl, device=device)
        val_loader = DeviceDataLoader(val_dl, device=device)    
        del train_data
        del val_data
        torch.cuda.empty_cache()
        for reg in regs:
            print("[Mode={:d}], [Train split:{:.2f}], [Alpha={:.2f}]" .format(mode,split,reg))
            # Dynamical Score Model 
            model = DyamicalScoreSDE_3D(
                                input_length=INPUT_SEQ, 
                                in_channels=1, 
                                residual_channels=layers,
                                marginal_prob_std_fn=marginal_prob_std_fn,
                                pred_var_dims=3,
                                scoreonly=scoreonly,
                                score_nlayers=score_layers,
                                reg=reg
                                ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            resume = args.resume
            if resume:
                checkpt = args.checkpt
                CHECKPT_EPOCH = checkpt
                CHECKPTFILE = "checkpt_"+str(mode)+"mode_alpha"+str(reg)+"_"+str(CHECKPT_EPOCH)+"_split"+str(split)+"_.pth"
                BESTCKPTFILE = "checkpt_best_"+str(mode)+"mode_alpha"+str(reg)+"_"+str(CHECKPT_EPOCH)+"_split"+str(split)+"_.pth"
                CHECKPTPATH = os.path.join(SAVEPATH,CHECKPTFILE)
                checkpoint = torch.load(CHECKPTPATH)
                model.load_state_dict(checkpoint)
                
            else:
                checkpt = 0
            
            run_model(model,
                train_loader, 
                val_loader, 
                optimizer, 
                split = split,
                n_mode=mode,
                reg=reg, 
                input_seq=INPUT_SEQ, 
                epochs=EPOCHS,
                checkpt=checkpt, 
                save=save, 
                scoreonly=scoreonly,
                folder=SAVEPATH
            )
        




