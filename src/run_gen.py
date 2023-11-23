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
WB_TRACK = config["paths"]["WB_TRACK"]
logging.basicConfig(level=logging.INFO)
from neural_ar.model import *
from utils import *
from data.dataset import *
from neural_ar import *
from gle_model.transient_gle import *
import wandb
import hydra 
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def eval_app(cfg: DictConfig) -> None:
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
        "nsamples": cfg.eval["nsamples"],
        "ngen": cfg.eval["ngen"]
        }
    )
    lr =  float(cfg.train["lr"])
    n_input = cfg.train["n_input"]
    temp = cfg.train["temp"]
    per_train =  cfg.train["per_train"]
    per_val = cfg.train["per_val"]
    temp = cfg.train["temp"]
    seed = cfg.train["seed"]
    checkpt = cfg.train["checkpt_epoch"]
    z_dim = cfg.train["z_dim"]
    nsamples = cfg.eval["nsamples"]
    ngen = int(float(cfg.eval["ngen"]))
    k = cfg.eval["k"]
    model = dynamical_model(
                        input_length=n_input, 
                        in_channels=1, 
                        residual_channels=z_dim,
                        pred_var_dims=3,
                        ).to(DEVICE)

    modes_traj = torch.tensor([nsamples, ngen, k, 3]).to(DEVICE)
    mode_corr = []
    genmode_corr = []
    for nmode in range(1,k):
        savepath = os.path.join(SAVEPATH, "T"+str(temp)+"/mode"+str(nmode)+"_"+str(n_input)+"stesps")
        gen_path = os.path.join(savepath, "gen")
        makedirs(gen_path)
        try:
            losspath = os.path.join(savepath, "metrics.csv")
        except OSError as e:
            print(e.errno)
        train_metrics = pd.read_csv(losspath)
        best_epoch = np.argmin(train_metrics["Val Loss"].values)+1
        #if nmode == 1:
        #    epoch = 24
        #print(epoch)
        sourcepath = os.path.join(DATAPATH, "ready/modesdata_T"+str(temp)+"_.pt")
        filepath = os.path.join(DATAPATH, "subtraj/subtraj_ds_T"+str(temp)+"_"+str(n_input)+"steps_mode"+str(nmode)+"_.pt")
        _, _, test_data, norms = subtraj_mode_dataset(
                                                sourcepath=sourcepath,
                                                filepath=filepath,
                                                n_input=n_input,
                                                split=[per_train, per_train + per_val*per_train], 
                                                nmode=nmode, 
                                                norm=True,
                                                device=DEVICE,
                                                test=True
                                                )
        
        CHECKPTFILE = "checkpt_epoch_"+str(best_epoch)+"_.pth"
        BESTCKPTFILE = "checkpt_best_.pth"
        CHECKPTPATH = os.path.join(SAVEPATH,CHECKPTFILE)
        checkpoint = torch.load(CHECKPTPATH)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Trained Autoregressive model embedded in an Implicit Euler Integrative Scheme
        mode_traj, mu_traj, sigma_traj  = EulerIntegrator( 
            model, 
            norms=norms,
            in_trajectories=test_data[:nsamples,:n_input],
            ngen=ngen,
            device=DEVICE,
            IN_CHANNELS=1,
            std=norms[0][nmode]
            )
        modes_traj[:,:,nmode] = mode_traj[0]
        
        #mode_corr.append(autocorrFFT(dataset_pos[:,:n_inpu ].swapaxes(1,2).contiguous().view(dataset.size(0)*3,-1)).mean(dim=0)/dataset_pos[:,:n_inpu].var())
        #genmode_corr.append(autocorrFFT(gen_data[:,:,0].swapaxes(1,2).contiguous().view(3*gen_data.size(0),-1)).mean(dim=0)/gen_data[:,:,0].var())
        torch.save(mode_traj, os.path.join(gen_path, "gen/gen_mode"+str(nmode)+"_epoch"+str(best_epoch)+"_.pt"))
        torch.save(mu_traj, os.path.join(gen_path, "gen/gen_mu_mode"+str(nmode)+"_epoch"+str(best_epoch)+"_.pt"))
        torch.save({"mu": mu_traj, "sigma": sigma_traj}, os.path.join(gen_path, "gen/gen_params_mode"+str(nmode)+"_epoch"+str(best_epoch)+"_.pt"))
    # Fit GLE model on short trajectories 
    gle_model = TransientGLE()
    gle_model.fit()
    params = gle_model.get_params()
    
    # Center of Mass dyamics with Transient GLE solution and generated modes 
    modes_traj[:,:,0] = gle_model(modes_traj)
    eval_metrics = {"Best Epoch": best_epoch,
                    "params": params,
                    }
    wandb.log(eval_metrics)
    
    
        
if __name__ == "__main__":
    eval_app()

        