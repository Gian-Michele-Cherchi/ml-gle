import torch 
from src.utils import *
from torch.distributions import MultivariateNormal

def EulerIntegrator(
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
    