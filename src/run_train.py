#########################################CONFIG 
import os
import yaml
import torch 
import logging
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
PROJECTPATH = config["paths"]["PROJECTPATH"]
DATAPATH = config["paths"]["DATAPATH"]
SAVEPATH = config["paths"]["SAVEPATH"]
DEVICE = config["paths"]["DEVICE"]
logging.basicConfig(level=logging.INFO)
from neural_ar.model import *
from utils import *
from data.dataset import *
from neural_ar import *
from torch.utils.data import DataLoader
import wandb
import hydra 
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def train_app(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project="ml-gle",
        config={
        "lr": float(cfg.train["lr"]),
        "architecture": cfg.train["architecture"],
        "dataset": cfg.train["dataset"],
        "n_input": cfg.train["n_input"],
        "epochs": cfg.train["epochs"],
        "per_train": cfg.train["per_train"],
        "per_val": cfg.train["per_val"],
        "temp": cfg.train["temp"], 
        "mode": cfg.train["mode"],
        "z_dim": cfg.train["z_dim"], 
        "train_batch": cfg.train["train_batch"],
        "val_batch": cfg.train["val_batch"],
        "activation": cfg.train["activation"],
        "seed": cfg.train["seed"],
        "save": cfg.train["save"],
        }
    )
    lr =  float(cfg.train["lr"])
    n_input = cfg.train["n_input"]
    temp = cfg.train["temp"]
    epochs= cfg.train["epochs"]
    per_train =  cfg.train["per_train"]
    per_val = cfg.train["per_val"]
    temp = cfg.train["temp"]
    mode = cfg.train["mode"]
    z_dim = cfg.train["z_dim"] 
    train_batch = cfg.train["train_batch"]
    val_batch = cfg.train["val_batch"]
    activation = cfg.train["activation"]
    seed = cfg.train["seed"]
    save = cfg.train["save"]
    checkpt = cfg.train["checkpt_epoch"]

    train_data, val_data, norms = subtraj_mode_dataset(
                                            sourcepath= os.path.join(DATAPATH, "ready/modesdata_T"+str(temp)+"_.pt"),
                                            filepath=os.path.join(DATAPATH, "subtraj/subtraj_ds_T"+str(temp)+"_"+str(n_input)+"steps_mode"+str(mode)+"_.pt"),
                                            n_input=n_input,
                                            split=[per_train, per_train + per_val*per_train], 
                                            nmode=mode, 
                                            norm=True,
                                            device=DEVICE
                                            )

    train_dl = DataLoader(train_data, train_batch, shuffle=True)
    val_dl = DataLoader(val_data, val_batch, shuffle=True)
    train_loader = DeviceDataLoader(train_dl, device=DEVICE)
    val_loader = DeviceDataLoader(val_dl, device=DEVICE)    

    print("[Mode={:d} NAR Training, input length "+str(n_input)+" steps]\n" .format(mode))
    # Dynamical Score Model 
    torch.manual_seed(seed)
    model = dynamical_model(
                    input_length=n_input, 
                    in_channels=1, 
                    residual_channels=4,
                    pred_var_dims=3,
                    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpt != "":
        print("[Loading state_dict from checkpoint epoch "+checkpt+"]\n" .format(mode))
        CHECKPT_EPOCH = int(checkpt)
        CHECKPTFILE = "checkpt_"+str(mode)+"mode_"+str(CHECKPT_EPOCH)+"_.pth"
        BESTCKPTFILE = "checkpt_best_"+str(mode)+"mode_"+str(CHECKPT_EPOCH)+"_.pth"
        CHECKPTPATH = os.path.join(SAVEPATH,CHECKPTFILE)
        checkpoint = torch.load(CHECKPTPATH)
        model.load_state_dict(checkpoint)

    else:
        checkpt = 0
    exp_folder = "mode"+str(mode)+"_"+str(n_input)+"steps"
    savepath = os.path.join(SAVEPATH, "T"+str(temp)+"/"+exp_folder)
    makedirs(savepath)
    run_model(model,
            train_loader, 
            val_loader, 
            optimizer,
            wandb,
            n_input=n_input, 
            epochs=epochs,
            checkpt=checkpt, 
            save=save, 
            exp_folder=savepath,
            )
            
    #print("[Loading state_dict from checkpoint epoch "+checkpt+"]\n" .format(mode))
    

if __name__ == "__main__":
    train_app()



