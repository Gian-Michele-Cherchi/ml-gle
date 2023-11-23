#########################################CONFIG 
import os
import yaml
import torch 
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
with open("src/config/train_conf.yaml") as f:
    train_config = yaml.load(f, Loader=yaml.FullLoader)
PROJECTPATH = config["PROJECTPATH"]
DATAPATH = config["READY_DATAPATH"]
SAVEPATH = config["SAVEPATH"]
DEVICE = config["DEVICE"]
logging.basicConfig(level=logging.INFO)
from neural_ar.model import *
from utils import *
from data.dataset import *
from neural_ar.training import *
from torch.utils.data import DataLoader
import logging
import wandb
import hydra 
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="train_conf")
def train_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project="ml-gle",
        config={
        "lr": float(train_config["LR"]),
        "architecture": train_config["ARCHITECTURE"],
        "dataset": train_config["DATASET"],
        "n_input": train_config["N_INPUT"],
        "epochs": train_config["EPOCHS"],
        "per_train": train_config["PER_TRAIN"],
        "per_val": train_config["PER_VAL"],
        "temp": train_config["TEMP"], 
        "mode": train_config["MODE"],
        "z_dim": train_config["Z_DIM"], 
        "train_batch": train_config["TRAIN_BATCH"],
        "val_batch": train_config["VAL_BATCH"],
        "activation": train_config["ACTIVATION"],
        "seed": train_config["SEED"],
        "save": train_config["SAVE"],
        }
    )
    lr =  float(train_config["LR"])
    n_input = train_config["N_INPUT"]
    temp = train_config["TEMP"]
    per_train = train_config["N_TRAIN"]
    epochs= train_config["EPOCHS"]
    per_train =  train_config["PER_TRAIN"]
    per_val = train_config["PER_VAL"]
    temp = train_config["TEMP"]
    mode = train_config["mode"]
    z_dim = train_config["Z_DIM"] 
    train_batch = train_config["TRAIN_BATCH"]
    val_batch = train_config["VAL_BATCH"]
    activation = train_config["ACTIVATION"]
    seed = train_config["SEED"]
    save = train_config["SAVE"]
    checkpt = train_config["CHECKPT_EPOCH"]

    train_data, val_data, _ ,norms = subtraj_mode_dataset(
                                            filepath="",
                                            n_input=n_input,
                                            split=[per_train, per_train + per_val*per_train], 
                                            nmode=mode, 
                                            norm=True,
                                            )

    train_dl = DataLoader(train_data, train_batch, shuffle=True)
    val_dl = DataLoader(val_data, val_batch, shuffle=True)
    train_loader = DeviceDataLoader(train_dl, device=DEVICE)
    val_loader = DeviceDataLoader(val_dl, device=DEVICE)    

    print("[Mode={:d} NAR Training, input length "+str(n_input)+" steps.]\n" .format(mode))
    # Dynamical Score Model 
    model = dynamical_model(
                    input_length=n_input, 
                    in_channels=1, 
                    residual_channels=4,
                    pred_var_dims=3,
                    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpt is not "":
        print("[Loading state_dict from checkpoint epoch "+checkpt+"]\n" .format(mode))
        CHECKPT_EPOCH = int(checkpt)
        CHECKPTFILE = "checkpt_"+str(mode)+"mode_"+str(CHECKPT_EPOCH)+"_.pth"
        BESTCKPTFILE = "checkpt_best_"+str(mode)+"mode_"+str(CHECKPT_EPOCH)+"_.pth"
        CHECKPTPATH = os.path.join(SAVEPATH,CHECKPTFILE)
        checkpoint = torch.load(CHECKPTPATH)
        model.load_state_dict(checkpoint)

    else:
        checkpt = 0

    run_model(model,
            train_loader, 
            val_loader, 
            optimizer, 
            split = per_train,
            n_mode=mode,
            n_input=n_input, 
            epochs=epochs,
            checkpt=checkpt, 
            save=save, 
            folder=SAVEPATH
            )
            
    print("[Loading state_dict from checkpoint epoch "+checkpt+"]\n" .format(mode))
    

if __name__ == "__main__":
    train_app()



