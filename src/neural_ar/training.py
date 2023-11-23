
#Config 
import os
import yaml
import sys 
import pandas as pd 
#############################################################
from neural_ar.training import *
from utils import *
import torch 
#import sys 
#from torch.nn.functional import mse_loss
#sys.path.append("/home/gmcherch/project")
#from library.autoencoders import *
#from utils import *
import numpy as np

BATCHLOSS_FREQ = 10


def evaluate_epoch(model, validation_loader, loss_function):
    batch_losses = [model.batch_validation(batch, loss_function) for batch in validation_loader]
    return model.validation_losses(batch_losses)

def train(epoch,
         train_loader,
         validation_loader, 
         model, 
         optimizer, 
         display_loss=True, 
         batch_losses=False
        ):

    train_loss = []
    valloss = []
    for index, train_data, in enumerate(train_loader):
        train_data = train_data.unsqueeze(1)
        optimizer.zero_grad()
        loss, _ = model(train_data)
        train_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        for index, val_data, in enumerate(validation_loader):
            val_data = val_data.unsqueeze(1)
            val_loss, _ = model(val_data)
            valloss.append(val_loss.cpu().item())
        

        if batch_losses:
            if index % BATCHLOSS_FREQ == 0:
                print('Epoch: {}, Batch [{}] | MSE Loss: {:.6f}] | MSE Val Loss = [{:.6f}]'
                .format(epoch, index, loss, val_loss))
            
    if display_loss:
        print('Epoch: [{}] | MSE Train Loss: [{:.6f}] | MSE Val Loss = [{:.6f}] '
        .format(epoch, np.mean(train_loss),np.mean(valloss)))
    
    return np.mean(train_loss), np.mean(valloss)



def train_dynamical_model(epoch,
                          input_seq,
                          train_loader,
                          validation_loader,
                          model, 
                          optimizer, 
                          display_loss=True, 
                          batch_losses=False
                            ):

    train_loss = []
    valloss = []
    for index, train_data in enumerate(train_loader):
        input, target = train_data[:,:1,:], train_data[:,1:,:]
        for t in range(input.size(2)-input_seq):
            optimizer.zero_grad()
            loss, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+input_seq]) 
            train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        for index, data in enumerate(validation_loader):
            input, target = data[:,:1,:], data[:,1:,:]
            for t in range(input.size(2)-input_seq):
                val_loss, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+input_seq])
                valloss.append(val_loss.cpu().item())
        

        if batch_losses:
            if index % BATCHLOSS_FREQ == 0:
                print('Epoch: {}, Batch [{}] | NLL Train: {:.6f}] | NLL Val  = [{:.6f}]'
                .format(epoch, index, loss, val_loss))
            
    if display_loss:
        print('Epoch: [{}] | NLL Train: [{:.6f}] | NLL Val: [{:.6f}] '
        .format(epoch, np.mean(train_loss),np.mean(valloss)))
    
    return np.mean(train_loss), np.mean(valloss)


def train_dynamical_model_2(epoch,
                          input_seq,
                          train_loader,
                          validation_loader,
                          model, 
                          optimizer, 
                          display_loss=True, 
                          batch_losses=False
                            ):

    train_loss = []
    valloss = []
    for index, train_data in enumerate(train_loader):
        input, target = train_data[:,:1,:], train_data[:,1:,:]
        T = input.size(2)
        break
    for t in range(T-input_seq):
        for index, train_data in enumerate(train_loader):
            input, target = train_data[:,:1,:], train_data[:,1:,:]
            optimizer.zero_grad()
            loss, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+input_seq]) 
            train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        for t in range(T-input_seq):
            for index, data in enumerate(validation_loader):
                input, target = data[:,:1,:], data[:,1:,:]
                val_loss, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+input_seq])
                valloss.append(val_loss.cpu().item())
        

        if batch_losses:
            if index % BATCHLOSS_FREQ == 0:
                print('Epoch: {}, Batch [{}] | NLL Train: {:.6f}] | NLL Val  = [{:.6f}]'
                .format(epoch, index, loss, val_loss))
            
    if display_loss:
        print('Epoch: [{}] | NLL Train: [{:.6f}] | NLL Val: [{:.6f}] '
        .format(epoch, np.mean(train_loss),np.mean(valloss)))
    
    return np.mean(train_loss), np.mean(valloss)
    


def train_recurrent_dynamical_model(epoch,
                            input_seq,
                            train_loader,
                            validation_loader,
                            model, 
                            optimizer, 
                            display_loss=True, 
                            batch_losses=False
                                ):

    train_loss = []
    valloss = []
   
    for index, train_data in enumerate(train_loader):
        input, target = train_data[:,:1,:], train_data[:,1:,:]
        batchsize = input.size(0)
        hidden_dims = model.hidden_dims
        h_t = torch.zeros(1,batchsize, hidden_dims).to(input)
        c_t = torch.zeros(1,batchsize, hidden_dims).to(input)
        for t in range(input.size(2)-input_seq):
            optimizer.zero_grad()
            loss, _, hidden_states = model(input[:,:,t:t+ input_seq],
                                           (h_t.detach(),c_t.detach()) ,
                                           target[:,:,t+input_seq]
                                           ) 
            h_t, c_t = hidden_states
            train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        for index, data in enumerate(validation_loader):
            input, target = data[:,:1,:], data[:,1:,:]
            batchsize = input.size(0)
            h_t = torch.zeros(1,batchsize, hidden_dims).to(input)
            c_t = torch.zeros(1,batchsize, hidden_dims).to(input)
            for t in range(input.size(2)-input_seq):
                val_loss, _, hidden_states = model(input[:,:,t:t+ input_seq],
                                           (h_t,c_t) ,
                                           target[:,:,t+input_seq]
                                           ) 
                h_t, c_t = hidden_states
                valloss.append(val_loss.cpu().item())
        

        if batch_losses:
            if index % BATCHLOSS_FREQ == 0:
                print('Epoch: {}, Batch [{}] | NLL Train: {:.6f}] | NLL Val  = [{:.6f}]'
                .format(epoch, index, loss, val_loss))
            
    if display_loss:
        print('Epoch: [{}] | NLL Train: [{:.6f}] | NLL Val: [{:.6f}] '
        .format(epoch, np.mean(train_loss),np.mean(valloss)))
    
    return np.mean(train_loss), np.mean(valloss)

def train_dynamical_score_sde(epoch,
                    input_seq,
                    train_loader,
                    validation_loader,
                    model, 
                    optimizer, 
                    display_loss=True, 
                    batch_losses=False,
                    score_only=False,
                    gauss_only=False,
                    ):
    
    train_loss = []
    val_loss = []
    train_score_loss = []
    train_dyn_loss = []
    val_score_loss = []
    val_dyn_loss = []
    avg_score_loss = 0.
    avg_dyn_loss = 0.
    num_items = 0
    for index, train_data in enumerate(train_loader):
        input, target = train_data[:,:1,:], train_data[:,1:,:]
        for t in range(input.size(2)-input_seq):
            optimizer.zero_grad()
            if score_only:
                tr_loss = model(input[:,:,t:t+ input_seq], target[:,:,t+input_seq], target[:,:,t+input_seq])
            elif gauss_only:
                tr_loss, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+input_seq], target[:,:,t+input_seq])
            else:
                loss, _, losses, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+ input_seq], target[:,:,t+input_seq])
                train_dyn_loss.append(losses[0].cpu().item())
                train_score_loss.append(losses[1].cpu().item())
                avg_score_loss += losses[0].item() * input.shape[0]
                avg_dyn_loss += losses[0].item() * input.shape[0]
                num_items += input.shape[0]
           
            train_loss.append(tr_loss.cpu().item())
            tr_loss.backward()
            optimizer.step()
            
        
    with torch.no_grad():
        for index, data in enumerate(validation_loader):
            input, target = data[:,:1,:], data[:,1:,:]
            for t in range(input.size(2)-input_seq):
                if score_only:
                    valloss = model(input[:,:,t:t+ input_seq], target[:,:,t+ input_seq], target[:,:,t+input_seq])
                elif gauss_only:
                    valloss, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+ input_seq], target[:,:,t+input_seq])
                else:
                    valloss, _, val_losses, _ = model(input[:,:,t:t+ input_seq], target[:,:,t+ input_seq], target[:,:,t+input_seq])
                    val_dyn_loss.append(val_losses[0].cpu().item())
                    val_score_loss.append(val_losses[1].cpu().item())
                val_loss.append(valloss.cpu().item())

        if batch_losses:
            if index % BATCHLOSS_FREQ == 0:
                print('Epoch: {}, Batch [{}] | NLL Train: {:.6f}] | NLL Val  = [{:.6f}]'
                .format(epoch, index, loss, val_loss))
                
    if score_only:
        if display_loss:
         print('Epoch: [{}] | Train Score: [{:.6f}] | Val Score: [{:.6f}] '
                            .format(epoch, np.mean(train_loss), np.mean(val_loss)))
    
        return np.mean(train_loss), np.mean(val_loss) 
    elif gauss_only:
        if display_loss:
         print('Epoch: [{}] | Train Loss: [{:.6f}] | Val Loss: [{:.6f}] '
                            .format(epoch, np.mean(train_loss), np.mean(val_loss)))
    
        return np.mean(train_loss), np.mean(val_loss) 
    
    else:
        if display_loss:
            print('Epoch: [{}] | Train Score: [{:.6f}] | Val Score: [{:.6f}| Train Dyn: [{:.6f}] | Val Dyn: [{:.6f}] '
            .format(epoch, np.mean(train_score_loss), np.mean(val_score_loss),np.mean(train_dyn_loss), np.mean(val_dyn_loss)))
        
        return np.mean(train_score_loss) , np.mean(train_dyn_loss), np.mean(val_score_loss), np.mean(val_dyn_loss), np.std(val_score_loss), np.std(val_dyn_loss)
    
    

def train_dynamical_score_sde_notseq(epoch,
                    train_loader,
                    validation_loader,
                    model, 
                    optimizer, 
                    input_seq,
                    display_loss=True, 
                    batch_losses=False,
                    score_only=False,
                    ):
    
    train_loss = []
    val_loss = []
    train_score_loss = []
    train_dyn_loss = []
    val_score_loss = []
    val_dyn_loss = []
    avg_score_loss = 0.
    avg_dyn_loss = 0.
    num_items = 0
    for index, train_data in enumerate(train_loader):
        input, target = train_data[:,:input_seq,:], train_data[:,input_seq:,:]
        
        optimizer.zero_grad()
        if score_only:
            tr_loss = model(input, target, target)
        else:
            tr_loss, _, losses, _ = model(input, target, target)
            train_dyn_loss.append(losses[0].cpu().item())
            train_score_loss.append(losses[1].cpu().item())
            avg_score_loss += losses[0].item() * input.shape[0]
            avg_dyn_loss += losses[0].item() * input.shape[0]
            num_items += input.shape[0]
        
        train_loss.append(tr_loss.cpu().item())
        tr_loss.backward()
        optimizer.step()
            
        
    with torch.no_grad():
        for index, data in enumerate(validation_loader):
            input, target = data[:,:input_seq,:], data[:,input_seq:,:]
            if score_only:
                valloss = model(input, target, target)
            
            else:
                valloss, _, val_losses, _ = model(input, target, target)
                val_dyn_loss.append(val_losses[0].cpu().item())
                val_score_loss.append(val_losses[1].cpu().item())
            val_loss.append(valloss.cpu().item())

        if batch_losses:
            if index % BATCHLOSS_FREQ == 0:
                print('Epoch: {}, Batch [{}] | NLL Train: {:.6f}] | NLL Val  = [{:.6f}]'
                .format(epoch, index, tr_loss, val_loss))
                
    if score_only:
        if display_loss:
         print('Epoch: [{}] | Train Score: [{:.6f}] | Val Score: [{:.6f}] '
                            .format(epoch, np.mean(train_loss), np.mean(val_loss)))
    
        return np.mean(train_loss), np.mean(val_loss) 
    else:
        if display_loss:
            print('Epoch: [{}] | Train Score: [{:.6f}] | Val Score: [{:.6f}| Train Dyn: [{:.6f}] | Val Dyn: [{:.6f}] '
            .format(epoch, np.mean(train_score_loss), np.mean(val_score_loss),np.mean(train_dyn_loss), np.mean(val_dyn_loss)))
        
        return np.mean(train_score_loss) , np.mean(train_dyn_loss), np.mean(val_score_loss), np.mean(val_dyn_loss), np.std(val_score_loss), np.std(val_dyn_loss)
    
 
 

BATCHLOSS_FREQ = 10
CHECKPT_FREQ = 1
   
    
    
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