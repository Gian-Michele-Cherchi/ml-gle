#Config 
import os
import pandas as pd 
from neural_ar.training import *
from utils import *
import torch 
import numpy as np
import logging
BATCHLOSS_FREQ = 10
CHECKPT_FREQ = 1

def training_step(epoch,
                train_loader,
                validation_loader,
                model, 
                optimizer, 
                input_seq:int,
                display_loss:bool=True, 
                batch_losses:bool=False,
                ):
    
    train_loss = []
    val_loss = []

    for index, train_data in enumerate(train_loader):
        input, target = train_data[:,:input_seq,:], train_data[:,input_seq:,:]
        optimizer.zero_grad()
        tr_loss, _, _ = model(input, target, target)
        train_loss.append(tr_loss.cpu().item())
        tr_loss.backward()
        optimizer.step()
            
     # validation   
    with torch.no_grad():
        for index, data in enumerate(validation_loader):
            input, target = data[:,:input_seq,:], data[:,input_seq:,:]
            valloss, _,  _ = model(input, target, target)
            val_loss.append(valloss.cpu().item())

        if batch_losses:
            if index % BATCHLOSS_FREQ == 0:
                print('Epoch: {}, Batch [{}] | NLL Train: {:.6f}] | NLL Val  = [{:.6f}]'
                .format(epoch, index, tr_loss, val_loss))
                
   
    if display_loss:
        print('Epoch: [{}] | Train Loss: [{:.6f}] | Val Loss: [{:.6f}] '
        .format(epoch, np.mean(train_loss), np.mean(val_loss)))
    
    return np.mean(train_loss), np.mean(val_loss), np.std(train_loss), np.std(val_loss)
   
    
    
def run_model(model,
              train_loader, 
              val_loader, 
              optimizer, 
              wandb,
              exp_folder,
              n_input:int=128, 
              epochs: int=100,
              checkpt: int=0, 
              save: bool=True,
              ):
    
    train_loss = []
    val_loss = []
    std_train_loss = []
    std_val_loss = []
    best_loss = np.inf
    #num_items = 0
    logging.info("[Train Start]")
    
    for epoch in range(checkpt +1 , epochs + checkpt +1):
        # TRAIN & VALIDATION 
        res  = training_step(epoch, 
                            train_loader, 
                            val_loader,
                            model, 
                            optimizer,
                            input_seq=n_input,
                            display_loss=True, 
                            batch_losses=False, 
                            )
        train_loss.append(res[0])
        val_loss.append(res[1])
        std_train_loss.append(res[2])
        std_val_loss.append(res[3])
        #nn.utils.clip_grad_norm_(model.parameters(), clip)
        wandb.log({"Train Loss": res[0], "Val Loss": res[1], "Train Std": res[2], "Val Std": res[3]})
        if save:   
            if not epoch % CHECKPT_FREQ: 
                
                filename = "checkpt_epoch_.pth"
                #makedirs(exp_folder)
                torch.save(model.state_dict(), os.path.join(exp_folder, filename))
            if res[1] < best_loss and res[1] >= res[0]:
                best_loss = res[1]
                filename = "checkpt_best_epoch"+str(epoch)+"_.pth"
                torch.save(model.state_dict(), os.path.join(exp_folder, filename))
                
            ####################################################################################   
    metrics = pd.DataFrame({'Train': train_loss,
                            'Val': val_loss, 
                            'Train std': std_train_loss,
                            "Val std": std_val_loss
                                    })
    
    if save:
        metrics.to_csv(os.path.join(exp_folder,"metrics.csv"))
        logging.info("[Losses file saved successfully]") 
    logging.info("[Train End]")
    
    return metrics