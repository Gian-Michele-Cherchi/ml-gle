
import numpy as np
import os
import pandas as pd
import yaml
import sys
import logging 
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PROJECTNAME = config["PROJECTNAME"]
USERPATH = config["USERPATH"]
SYSTEM = config["SYSTEM"]
FULLPATH  = os.path.join(USERPATH, PROJECTNAME)
SUBMODULE = os.path.join(FULLPATH, "ia")
sys.path.append(SUBMODULE)
BATCHLOSS_FREQ = 10
CHECKPT_FREQ = 1
#############################################################
from ia.library.training import *
from ia.library.utils import *

def run_model(model,
              train_loader, 
              val_loader, 
              optimizer, 
              n_mode,
              reg, 
              split,
              folder,
              input_seq:int=256, 
              epochs: int=50,
              checkpt: int=0, 
              save: bool=True, 
              scoreonly: bool=False,
              ):
    train_loss = []
    val_loss = []
    train_score_loss = []
    val_dyn_loss = []
    train_dyn_loss = []
    val_score_loss = []
    val_score_std = []
    val_dyn_std = []
    best_loss = np.inf
    avg_loss = 0.
    num_items = 0
    #logging.info("[Training Started]")
    exp_folder = folder
    for epoch in range(checkpt +1 , epochs + checkpt +1):
        # TRAIN & VALIDATION 
        res  = train_dynamical_score_sde_notseq(epoch, 
                                        train_loader, 
                                        val_loader,
                                        model, 
                                        optimizer,
                                        input_seq=input_seq,
                                        display_loss=True, 
                                        batch_losses=False, 
                                        score_only=scoreonly,
                                        )
        
        if scoreonly:
            loss_train, loss_val = res[0], res[1]
            train_loss.append(res[0])
            val_loss.append(res[1])
            
        
        else:
            train_score_loss.append(res[0])
            train_dyn_loss.append(res[1])
            val_score_loss.append(res[2])
            val_dyn_loss.append(res[3])
            val_score_std.append(res[4])
            val_dyn_std.append(res[5])

        #nn.utils.clip_grad_norm_(model.parameters(), clip)
            loss_train = res[0] + res[1]
            loss_val = res[2] + res[3]
        if save:
            makedirs("save")     #create save folder if not existent 
            if not epoch % CHECKPT_FREQ: 
                if scoreonly:
                   
                    filename = "checkpt_scoreonly"+str(n_mode)+"mode_"+str(epoch)+"_split"+str(split)+"_.pth"
                    #makedirs(exp_folder)
                    torch.save(model.state_dict(), os.path.join(exp_folder, filename))
                else:
                    filename = "checkpt_"+str(n_mode)+"mode_alpha"+str(reg)+"_"+str(epoch)+"_split"+str(split)+"_.pth"
                    #makedirs(exp_folder)
                    torch.save(model.state_dict(), os.path.join(exp_folder, filename))
            if loss_val < best_loss and loss_val >= loss_train:
                best_loss = loss_val
                if scoreonly:
                    filename = "checkpt_best_scoreonly"+str(n_mode)+"mode_split"+str(split)+"_.pth"
                    torch.save(model.state_dict(), os.path.join(exp_folder, filename))
                else:
                    filename = "checkpt_best_"+str(n_mode)+"mode_alpha"+str(reg)+"_split"+str(split)+"_.pth"
                    torch.save(model.state_dict(), os.path.join(exp_folder, filename))
                
    ####################################################################################      
    if scoreonly:
        losses = pd.DataFrame({'Train Loss': train_loss,
                            'Val Loss': val_loss, 
                            })
        losses.to_csv(os.path.join(exp_folder,"losses_scoreonly_"+str(n_mode)+"modes_split"+str(split)+"_.csv"))
        
    else:
        losses = pd.DataFrame({'Train Score': train_score_loss,
                                'Train Dyn': train_dyn_loss,
                                'Val Score': val_score_loss, 
                                'Val Dyn': val_dyn_loss, 
                                'Val Score Std': val_score_std,
                                'Val Dyn Std': val_dyn_std
                                })
        
        losses.to_csv(os.path.join(exp_folder,"losses_"+str(n_mode)+"modes_alpha"+str(reg)+"_split"+str(split)+"_.csv"))