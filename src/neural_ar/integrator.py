import torch 
from utils import *
from torch.distributions import MultivariateNormal
import tqdm
def EulerIntegrator(
                model, 
                norms,
                in_trajectories, 
                ngen: int,
                device: str, 
                IN_CHANNELS: int=1,
                **kwargs
                ):
    #std = kwargs["std"]
    #torch.manual_seed(seed)
    INPUT_SEQ = in_trajectories.size(1)
    NPOL = in_trajectories.size(0)
    gen_data = torch.zeros(NPOL, ngen+INPUT_SEQ, IN_CHANNELS+1, 3, device=device)
    mu_gen = torch.ones(NPOL ,ngen,IN_CHANNELS,3, device=device)
    sigma_gen = torch.zeros(NPOL, ngen,IN_CHANNELS,3,3, device=device)
    
    gen_data[:,:INPUT_SEQ,0] = in_trajectories
    open_log = tqdm.tqdm(total=0, position=0, bar_format='{desc}')
    open_log.set_description_str(
        f"Implicit Euler Integration with "+str(ngen)+" steps", 
        )
    progress_bar = tqdm.tqdm(total=ngen, desc="Advancement", position=1)
    with torch.no_grad():
        for t in range(ngen):
        
            rot_mean_t, _  = model.get_parameters(gen_data[:,t:t+ INPUT_SEQ,0])
           
            rot_matrix = model.rotation_matrix
            sigma_t = model.sigma 
            mu_gen[:,t,:] = torch.bmm(rot_mean_t[:,None,...], rot_matrix.transpose(1,2))
            sigma_gen[:,t,0,:] = sigma_t
            
            #average velocity sampling conditioned on historic trajectory, in the rotated ref. frame 
            rot_trial_sample = MultivariateNormal(rot_mean_t, sigma_t).sample().to(device)
            trial_sample = torch.bmm(rot_trial_sample[:,None,...], rot_matrix.transpose(1,2))
            gen_data[:,INPUT_SEQ +t,1]  = trial_sample[:,0,:] #sampled velocity
            gen_data[:,INPUT_SEQ +t,0]  = gen_data[:,t+ INPUT_SEQ-1,0]  + norms*trial_sample[:,0,:]  # integrated position
            progress_bar.update(1)
    return gen_data, mu_gen, sigma_gen
    