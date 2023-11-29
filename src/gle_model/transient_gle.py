
import torch
import os
from dataclasses import dataclass
from utils import msd_fft
from sklearn.linear_model import LinearRegression

@dataclass
class TransientGLE():
    """
        Generalized Langevin Equation class modelling transient anomalous diffusion in Polymer Melts 
        with single-polymer center of mass and Normal modes dynamics. 
    """
    
   
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the soluton to the Transient GLE, given modes' trajectories as input and fitted parameters. 
        
        INPUT: 
            - z [batchsize, nframes, nmodes,3]: Normal modes trajectories either from MD or Generated 
        OUTPUT 
            - x [batchsize, nframes, 3]: Position Dynamics following GLE fitted solution
        """
        if self.reg is None:
            raise ValueError("Paremeters not found. Call fit() first.")
        
        self.nsamples = z.size(0)
        self.nframes = z.size(1)
        self.alpha = torch.sqrt(self.params)[0][0]
        self.beta = torch.sqrt(self.params)[0][1]
        bm = torch.randn(self.nsamples, self.nframes, 3).to(z).cumsum(dim=2) # Standard Brownian Motion 
        x = self.alpha*bm + self.beta*z.sum(dim=2)
        return x
    
    def fit(self, 
            z:  torch.Tensor, 
            n_max: int,
            n_min: int=1,
            n_points: int=3 
        ) ->  torch.Tensor:
        """
        Fit parameters to modes and center of mass data. 
        
        INPUT 
            - x [batchsize, nframes, nmodes,3]: Position trajectories 
            - z [batchsize, nframes, nmodes,3]: modes' trajectories
            - n_points: number of regression points 

        OUTPUT:
        """
        

        self.nsamples = z.size(0)
        self.nframes = z.size(1)
        k = z.size(2)
        assert k>1, ValueError("The number of modes should be at least one.")
        #self.params = torch.zeros(nmodes,2)
        
        modes = z[:,:,1:].swapaxes(2,3).swapaxes(1,2).contiguous().view(self.nsamples*3,self.nframes,k-1)
        
        # linear brownian variance term 
        t = torch.arange(n_min,self.nframes).unsqueeze(-1).to(z)
        f_tau = torch.zeros(self.nframes-n_min, k-1).to(z)
        
        for tau in range(1,self.nframes):
            #f_tau[:,tau-bal_end,:] = (z[:,:NMAX_MODES-(tau)]- z[:,(tau):])[:,0,:]**2
            f_tau[tau-n_min,:] = ((modes[:,:self.nframes-(tau)]- modes[:,(tau):])**2).mean(dim=(0,1))
        x_msd = msd_fft(z[:,:,0].swapaxes(1,2).contiguous().view(z.size(0)*3,-1)).mean(dim=0)[n_min:]
        
        
        reg_point_index = torch.linspace(0,n_max, n_points, dtype=int)
        self.reg = LinearRegression(positive=True)
        
       
        f = torch.cat([t[reg_point_index], f_tau[reg_point_index].sum(dim=1, keepdim=True)], dim=1)
        reg = self.reg.fit(f.cpu().numpy(), x_msd[reg_point_index,None].cpu().numpy())
        self.params = torch.tensor(reg.coef_)
    
    def get_params(self) -> torch.Tensor:
        if self.reg is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.params
    
    


    
    
    

        
        
    
        
        
        
        