#########################################CONFIG 
import os
import yaml
import torch 
import logging
logging.basicConfig(level=logging.INFO)
from neural_ar.model import *
from utils import *
from data.dataset import *
from neural_ar import *
from gle_model.transient_gle import *
import wandb
import hydra 
from omegaconf import DictConfig, OmegaConf

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader) 
PROJECTPATH = config["paths"]["PROJECTPATH"]
DATAPATH = config["paths"]["DATAPATH"]
SAVEPATH = config["paths"]["SAVEPATH"]
DEVICE = config["paths"]["DEVICE"]


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
        "ngen": cfg.eval["ngen"], 
        "k": cfg.eval["k"]
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
    flag_acf = cfg.eval["acf"]
    flag_gle = cfg.eval["gle"]
    n_train = int(per_train*10000)
    # NAR model 
    model = dynamical_model(
                        input_length=n_input, 
                        in_channels=1, 
                        residual_channels=z_dim,
                        pred_var_dims=3,
                        ).to(DEVICE)

    modes_traj = torch.Tensor(nsamples, ngen+n_input, k, 3).to(DEVICE)
    gen_nacfs = {}
    sourcepath = os.path.join(DATAPATH, "ready/modesdata_T"+str(temp)+"_.pt")
    test_data = torch.load(sourcepath, map_location=DEVICE).swapaxes(0,1)
    norms = [torch.sqrt((test_data[:,:n_train]**2).mean(dim=(0,1,3))),torch.sqrt((test_data[:,:n_train].diff(dim=1)**2).mean(dim=(0,1,3)))]
    
    for nmode in range(1,k):
        
        savepath = os.path.join(SAVEPATH, "T"+str(temp)+"/mode"+str(nmode)+"_"+str(n_input)+"steps")
        gen_path = os.path.join(savepath, "gen")
        makedirs(gen_path)
        
        try:
            losspath = os.path.join(savepath, "metrics.csv")
            train_metrics = pd.read_csv(losspath)
        except OSError as e:
            print(e.errno)
        
        best_epoch = np.argmin(train_metrics["Val"].values)+1
        #if nmode == 1:
        #    best_epoch = 24
        #if nmode == 2:
        #    best_epoch = 21
            #print(epoch)
        mode_filepath = os.path.join(gen_path, "gen_mode_epoch"+str(best_epoch)+"_.pt")
        try: 
            # If file exists, load in memory instead of generating 
            modes_traj[:, :, nmode] = torch.load(mode_filepath, map_location=DEVICE)[:,:ngen,0]
        # Generate mode dynamics 
        except OSError as e:
            print("[File not Found. Starting Mode "+str(nmode)+" Autoregressive Generation with model at epoch "+str(best_epoch)+"]\n")
            #test_data, norms = subtraj_mode_dataset(
            #                                    sourcepath=sourcepath,
            #                                    filepath=filepath,
            #                                    n_input=n_input,
            #                                    split=[per_train, per_train + per_val*per_train], 
            #                                    nmode=nmode, 
            #                                    norm=True,
            #                                    device=DEVICE,
            #                                    test=True
            #                                    )
            
            # Load model weights 
            checkpt_file = "checkpt_epoch_"+str(best_epoch)+".pth"
            #BESTCKPTFILE = "checkpt_best_.pth"
            checkpt_path = os.path.join(savepath,checkpt_file)
            checkpoint = torch.load(checkpt_path)
            model.load_state_dict(checkpoint)
            model.eval()
            
            # Trained Autoregressive model embedded in an Implicit Euler Integrative Scheme
            mode_traj, mu_traj, sigma_traj  = EulerIntegrator( 
                                                        model, 
                                                        norms=norms[1][nmode]/norms[0][nmode],
                                                        in_trajectories=test_data[:,n_train: n_train + n_input,nmode],
                                                        ngen=ngen,
                                                        device=DEVICE,
                                                        IN_CHANNELS=1,
                                                        )
            modes_traj[:,:,nmode] = mode_traj[:,:,0]
            if flag_acf:
                tmp = {}
                acf = autocorrFFT(mode_traj[:,:,0].swapaxes(1,2).contiguous().view(mode_traj[:,:,0].size(0)*3,-1))/mode_traj[:,:,0].var()
                tmp["nacfs"] = acf.mean(dim=0)
                tmp["SE"] = torch.tensor(list(map(lambda i: acf_se(acf[:,:i],acf.size(1)), range(1,acf.size(1)))))
                gen_nacfs["mode"+str(nmode)] = tmp   
                torch.save(os.path.join(gen_path, "gen_mode_nacfs_T"+str(temp)+"_ngen"+str(ngen)+"_.pt"))
                
            torch.save(mode_traj,mode_filepath)
            torch.save({"mu": mu_traj, "sigma": sigma_traj}, os.path.join(gen_path, "gen_params_mode"+str(nmode)+"_epoch"+str(best_epoch)+"_.pt"))
            
            eval_metrics = {"mode": nmode,
                            "Best Epoch": best_epoch,
                            "params": params,
                            }
            wandb.log(eval_metrics)
            pd.save_csv(os.path.join(gen_path,"eval_metrics.csv"))
    if flag_gle: 
        # Fit GLE model on short trajectories 
        gle_model = TransientGLE()
        gle_model.fit()
        params = gle_model.get_params()
        
        # Center of Mass dyamics with Transient GLE solution and generated modes 
        modes_traj[:,:,0] = gle_model(modes_traj)
    
        
       
if __name__ == "__main__":
    eval_app()

        