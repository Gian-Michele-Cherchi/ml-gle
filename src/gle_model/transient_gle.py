
import torch
import os
from dataclasses import dataclass
#from dataclasses import field 
from typing import List
from src.utils import msd_fft
from sklearn.linear_model import LinearRegression

@dataclass 
class TransientGLE():
    """
        Generalized Langevin Equation class modelling transient anomalous diffusion in Polymer Melts 
        with single-polymer center of mass and Normal modes dynamics. 
    """
     
    alpha: float
    beta: float
    nsamples: int 
    nframes: int
    model: object=None
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the soluton to the Transient GLE, given modes' trajectories as input and fitted parameters. 
        
        INPUT: 
            - z [batchsize, nframes, nmodes,3]: Normal modes trajectories either from MD or Generated 
        OUTPUT 
            - x_gle [batchsize, nframes, 3]: Position Dynamics following GLE fitted solution
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        self.nsamples = z.size(0)
        self.nframes = z.size(1)
        
        bm = torch.randn(self.nsamples, self.nframes, 3).to(z).cumsum(dim=2) # Wiener Process 
        x_gle = self.alpha*bm + self.beta*z.sum(dim=2)
        return x_gle
    
    def fit(self, 
            x: torch.Tensor, 
            z:  torch.Tensor, 
            n_points: int, 
            t_max: int,
            t_min: int=None) ->  torch.Tensor:
        """
        Fit parameters to modes and center of mass data. 
        
        INPUT 
            - x [batchsize, nframes, nmodes,3]: Position trajectories 
            - z [batchsize, nframes, nmodes,3]: modes' trajectories
            - n_points: number of regression points 
            - t_max: regression window's upper bound 
            - t_min: regression window's lower bound 

        OUTPUT:
        """
        
        if t_min is None:
            t_min = 1
        self.nsamples = z.size(0)
        self.nframes = z.size(1)
        nmodes = z.size(2)
        self.params = torch.zeros(nmodes,2)
        
        z = z.swapaxes(2,3).swapaxes(1,2).contiguous().view(self.samples*3,self.nframes,nmodes)
        
        # linear brownian variance term 
        t = torch.arange(1,self.nframes).unsqueeze(-1).to(z)
        f_tau = torch.zeros(self.nframes-t_min, nmodes).to(z)
        
        for tau in range(1,self.nframes):
            #f_tau[:,tau-bal_end,:] = (z[:,:NMAX_MODES-(tau)]- z[:,(tau):])[:,0,:]**2
            f_tau[tau-1,:] = ((z[:,:self.nframes-(tau)]- z[:,(tau):])**2).mean(dim=(0,1))
        x_msd = msd_fft(x.swapaxes(1,2).contiguous().view(x.size(0)*3,-1)).mean(dim=0)[1:]
        
        
        reg_points = torch.linsapce(t_min, t_max, n_points, dtype=int)
        self.expl_variance = torch.tensor(nmodes,1)
        self.model = LinearRegression(positive=True)
        for mod in range(nmodes):
            f = torch.cat([t[reg_points], f_tau[reg_points,:mod+1].sum(dim=1, keepdim=True)], dim=1)
            reg = self.model.fit(f.cpu().numpy(), x_msd[reg_points,None].cpu().numpy())
            self.params[mod,:] = torch.tensor(reg.coef_)
            self.expl_variance[mod] = self.params[:,0] + self.params[:,1]*f_tau[reg_points,:mod+1].sum(dim=1, keepdim=True)
        
    def get_expl_variance(self) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.expl_variance
    
    def get_params(self) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.params
    


    
    
    

        
        
    
        
        
        
        