import torch 
from library.utils import *
from torch.distributions import MultivariateNormal
from .num_sde_solver import Conditional_EM_Sampler
from scipy.stats import norm

def DynamicsIntegration_new(
                model, 
                norms,
                in_trajectories, 
                NGENSTEP: int,
                device: str, 
                IN_CHANNELS: int=1,
                sampling: str=None,  
                **kwargs
                ):
    std = kwargs["std"]
    #torch.manual_seed(seed)
    INPUT_SEQ = in_trajectories.size(1)
    NPOL = in_trajectories.size(0)
    gen_data = torch.zeros(NPOL, NGENSTEP+INPUT_SEQ, IN_CHANNELS+1, 3, device=device)
    mu_gen = torch.ones(NPOL ,NGENSTEP,IN_CHANNELS,3, device=device)
    sigma_gen = torch.zeros(NPOL, NGENSTEP,IN_CHANNELS,3,3, device=device)
    
    gen_data[:,:INPUT_SEQ,0] = in_trajectories
    with torch.no_grad():
        for t in range(NGENSTEP):
        
            rot_mean_t, _  = model.get_parameters(gen_data[:,t:t+ INPUT_SEQ,0])
           
            rot_matrix = model.rotation_matrix
            sigma_t = model.sigma 
            mu_gen[:,t,:] = torch.bmm(rot_mean_t[:,None,...], rot_matrix.transpose(1,2))
            sigma_gen[:,t,0,:] = sigma_t
            
            
            if sampling == "metropolis":
                #cond_mean = torch.bmm(rot_mean_t[:,None,...], rot_matrix.transpose(1,2))
                #x_start = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*cond_mean
                flag = False
                while flag == False:
                    #average velocity sampling conditioned on historic trajectory, in the rotated ref. frame 
                    rot_trial_sample = MultivariateNormal(rot_mean_t, sigma_t).sample().to(device)
                    trial_sample = torch.bmm(rot_trial_sample[:,None,...], rot_matrix.transpose(1,2))
                    x_proposal = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*trial_sample[:,0,:]  # integrated position
                    alpha = torch.exp(-(1/2)*(x_proposal**2))
                    u = torch.rand([NPOL,3]).to(device)
                    if (torch.sum(u <= alpha,dim=1) >= 2*torch.ones(NPOL,dtype=int,device=device)).sum() == 2*NPOL :
                         #print(alpha.mean(), alpha.std())
                         gen_data[:,INPUT_SEQ +t,1]  = trial_sample[:,0,:] #sampled velocity
                         gen_data[:,INPUT_SEQ +t,0]  = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*trial_sample[:,0,:]  # integrated position
                         flag = True
                       
            else:
                #average velocity sampling conditioned on historic trajectory, in the rotated ref. frame 
                rot_trial_sample = MultivariateNormal(rot_mean_t, sigma_t).sample().to(device)
                trial_sample = torch.bmm(rot_trial_sample[:,None,...], rot_matrix.transpose(1,2))
                gen_data[:,INPUT_SEQ +t,1]  = trial_sample[:,0,:] #sampled velocity
                gen_data[:,INPUT_SEQ +t,0]  = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*trial_sample[:,0,:]  # integrated position
                

    return gen_data, mu_gen, sigma_gen


def DynamicsIntegration_old(
                model, 
                norms,
                in_trajectories, 
                NGENSTEP: int,
                device: str, 
                IN_CHANNELS: int=1, 
                score_gen: bool=False,
                **kwargs
                ):
    #seed = kwargs["seed"] 
#manual seed
    #torch.manual_seed(seed)
    INPUT_SEQ = in_trajectories.size(1)
    NPOL = in_trajectories.size(0)
    gen_data = torch.zeros(NPOL, NGENSTEP+INPUT_SEQ, IN_CHANNELS+1, 3, device=device)
    mu_gen = torch.ones(NPOL ,NGENSTEP,IN_CHANNELS,3, device=device)
    sigma_gen = torch.zeros(NPOL, NGENSTEP,IN_CHANNELS,3,3, device=device)
    score_ev = []
    x_ev = []
    gen_data[:,:INPUT_SEQ,0] = in_trajectories
    with torch.no_grad():
        for t in range(NGENSTEP):
        
            rot_mean_t, _  = model.get_parameters(gen_data[:,t:t+ INPUT_SEQ,0])
           
            rot_matrix = model.rotation_matrix
            sigma_t = model.sigma 
            mu_gen[:,t,:] = torch.bmm(rot_mean_t[:,None,...], rot_matrix.transpose(1,2))
            sigma_gen[:,t,0,:] = sigma_t
            
            #average velocity sampling conditioned on historic trajectory, in the rotated ref. frame 
            rot_trial_sample = MultivariateNormal(rot_mean_t, sigma_t).sample().to(device)
            if score_gen:
               
                #rot_trial_x = torch.bmm(rot_matrix.transpose(1,2),gen_data[:,INPUT_SEQ +t,:1].swapaxes(1,2)).swapaxes(1,2)
                #score_sampler = kwargs["score_sampler"]
                marginal_prob_std_fn = kwargs["marginal_prob_std"]
                diffusion_coeff_fn = kwargs["diffusion_coeff"]
                score_gen_pos, x_dyn, score_dyn = Conditional_EM_Sampler(model, 
                                                                    [rot_trial_sample.unsqueeze(1), gen_data[:,t:t+ INPUT_SEQ,:1]], 
                                                                    marginal_prob_std_fn, 
                                                                    diffusion_coeff_fn, 
                                                                    num_steps=kwargs["score_steps"], 
                                                                    eps=1e-3,
                                                                    score_only=False,
                                                                    #snr=.16
                                                                  )
                score_ev.append(score_dyn)
                x_ev.append(x_dyn)
                gen_data[:,INPUT_SEQ +t,1:] = torch.bmm(score_gen_pos, rot_matrix.transpose(1,2))
               
                gen_data[:,INPUT_SEQ +t,0]  = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*gen_data[:,INPUT_SEQ +t,1]  # integrated position
                #x_dyn_corr = torch.mul(gen_data[:,INPUT_SEQ +t,:1].expand(-1, INPUT_SEQ,3),gen_data[:,t:INPUT_SEQ +t,0].flip(dims=[1])).mean(dim=(0,2))
            else:
              
                trial_sample = torch.bmm(rot_trial_sample[:,None,...], rot_matrix.transpose(1,2))
               
                gen_data[:,INPUT_SEQ +t,1]  = trial_sample[:,0,:] #sampled velocity
                gen_data[:,INPUT_SEQ +t,0]  = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*trial_sample[:,0,:]  # integrated position
                 
                
                #x_dyn_corr = torch.mul(gen_data[:,INPUT_SEQ +t,:1].expand(-1, INPUT_SEQ,3),gen_data[:,t:INPUT_SEQ +t,0].flip(dims=[1])).mean(dim=(0,2))

    return gen_data, mu_gen, sigma_gen, score_ev, x_ev


def ScoreDynamicsIntegration(
                model, 
                norms,
                in_trajectories, 
                NGENSTEP: int,
                device: str, 
                IN_CHANNELS: int=1, 
                gen_x: bool=False,
                **kwargs
                ):
    #seed = kwargs["seed"] 
#manual seed
    #torch.manual_seed(seed)
    INPUT_SEQ = in_trajectories.size(1)
    NPOL = in_trajectories.size(0)
    if gen_x:
        gen_data = torch.zeros(NPOL, NGENSTEP+INPUT_SEQ, IN_CHANNELS+1, 3, device=device)
    else:
        gen_data = torch.zeros(NPOL, NGENSTEP+INPUT_SEQ, IN_CHANNELS+1, 3, device=device)
    score_ev = torch.ones(NPOL ,NGENSTEP,kwargs["score_steps"],3, device=device)
    x_ev = torch.zeros(NPOL, NGENSTEP,kwargs["score_steps"],3, device=device)
   
    gen_data[:,:INPUT_SEQ,0] = in_trajectories
    with torch.no_grad():
        checkpt_bar = tqdm.tqdm(total=NGENSTEP, desc="ScoreGen", position=0)
        for t in range(NGENSTEP):
        
            
            #rot_trial_x = torch.bmm(rot_matrix.transpose(1,2),gen_data[:,INPUT_SEQ +t,:1].swapaxes(1,2)).swapaxes(1,2)
            #score_sampler = kwargs["score_sampler"]
            marginal_prob_std_fn = kwargs["marginal_prob_std"]
            diffusion_coeff_fn = kwargs["diffusion_coeff"]
            rot_traj = model.Rotate(gen_data[:,t:t+ INPUT_SEQ,0,:])
            score_gen_pos, x_dyn, score_dyn = Conditional_EM_Sampler(model, 
                                                                rot_traj, 
                                                                marginal_prob_std_fn, 
                                                                diffusion_coeff_fn, 
                                                                num_steps=kwargs["score_steps"], 
                                                                eps=1e-3,
                                                                score_only=True,
                                                                #snr=.16
                                                                )
            score_ev[:,t,:,:] = score_dyn
            x_ev[:,t,:,:] = x_dyn
            if gen_x:
                rot_matrix = model.rotation_matrix
                gen_data[:,INPUT_SEQ +t,:1] = torch.bmm(score_gen_pos, rot_matrix.transpose(1,2))
            else:
                rot_matrix = model.rotation_matrix
                gen_data[:,INPUT_SEQ +t,1:] = torch.bmm(score_gen_pos, rot_matrix.transpose(1,2))
                gen_data[:,INPUT_SEQ +t,0]  = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*gen_data[:,INPUT_SEQ +t,1]  # integrated position
            checkpt_bar.update(1)
    return gen_data, score_ev, x_ev
    
    
    
    