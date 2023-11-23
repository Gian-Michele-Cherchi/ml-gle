import os

import torch.nn as nn 
from utils import * 




class dynamical_model(nn.Module):
    
    def __init__(self, 
                input_length: int,
                in_channels: int,
                residual_channels: int, 
                pred_var_dims: int=3,
                activation: str='relu',
                diagonal: bool=False,
                ):
        super(dynamical_model, self).__init__()
        self.in_channels = in_channels
        self.input_feat = 3*in_channels*input_length 
        self.pred_var_dims = pred_var_dims
        self.diagonal = diagonal
    
        bias = True
     
        assert self.input_feat//8 > self.res_channels*3
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_feat, out_features=self.input_feat//2, bias=bias),
            activation_func(activation),
            #nn.Linear(in_features=self.input_feat, out_features=self.input_feat//2, bias=bias),  
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//2, out_features=self.input_feat//4, bias=bias),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//4, out_features=self.input_feat//8, bias=bias),
            #activation_func(activation),
            nn.Linear(in_features=self.input_feat//2, out_features=self.res_channels*3),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//2, out_features=self.input_feat//4),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//4, out_features=self.input_feat//8),
            #activation_func(activation), 
        )
        self.dropout = nn.Dropout(0.5)
       
        
        
        self.enc_to_mean = nn.Linear(self.res_channels*3, self.pred_var_dims)
        if self.diagonal:
            self.enc_to_var = nn.Linear(self.res_channels*3, self.pred_var_dims)
        else:
            self.alpha = int(self.pred_var_dims*(1+ (self.pred_var_dims-1)/2)) # n*(n-1)/2
            self.enc_to_var = nn.Linear(self.res_channels*3, self.alpha)
       
            
    def forward(self, history_input, target_x, target_vel):
        #INPUT: batchsize x 1 x k history x nmodes x 3 
        #TARGET: batchsize x 1 x nmodes x 3 
        # encoding
        #self.history_input = history_input
        #self.history_input = history_input.squeeze(1).squeeze(2)
        self.rot_history = self.Rotate(history_input)
        out = self.encoder(self.rot_history.view( self.rot_history.size(0),-1))
        
        out_drop = self.dropout(out)
        rot_mean, logvar, encoded = self.encode(out_drop)
        
        #target_x = target_x.squeeze(1)
        #target_vel = target_vel.squeeze(1)
        rot_target_vel = torch.bmm(self.rotation_matrix.transpose(1,2),target_vel.swapaxes(1,2)).squeeze(-1) #target rotation to frame (t-1,t-2,t-3)
        rot_target_x = torch.bmm(self.rotation_matrix.transpose(1,2),target_x.swapaxes(1,2)).swapaxes(1,2)

        # Loss evaluation 
        dyn_loss = self.NLL(rot_mean, logvar, rot_target_vel)
        loss = dyn_loss 
        return loss, [rot_mean,logvar], dyn_loss, encoded 
               
    
    def encode(self, out): 
        
        means = self.enc_to_mean(out)
        if self.diagonal:
           logvarsquared = self.enc_to_var(out)
           return means, logvarsquared
        else:
            elements = self.enc_to_var(out)
            lower = torch.zeros(elements.size(0), self.pred_var_dims, self.pred_var_dims).to(elements)
            batched_eye = torch.eye(self.pred_var_dims).unsqueeze(0).expand(elements.size(0),-1,-1).to(elements)
            lower = lower + batched_eye
            index = np.cumsum([n for n in range(0,self.pred_var_dims)])
            for row in range(1,self.pred_var_dims):
                lower[:,row,:row] = elements[:,index[row-1]:index[row]]

            diag = torch.exp(elements[:,self.alpha-self.pred_var_dims:]).unsqueeze(1)*batched_eye
            partial_prod1 = torch.bmm(lower, diag)
            self.sigma = torch.bmm(partial_prod1, lower.transpose(1,2))
            return means, [diag, lower], out
    
    def NLL(self, mean,logvar, target, diagonal=False):
        diag, lower = logvar
        partial_prod1 = torch.bmm(lower, diag)
        self.sigma = torch.bmm(partial_prod1, lower.transpose(1,2))
        sigma_inverse = torch.inverse(self.sigma)
        partial_prod2  = torch.bmm((mean-target).unsqueeze(1), sigma_inverse)
        logprob = -(1/2)*torch.diagonal(diag, dim1=1, dim2=2).sum(dim=1) - torch.bmm(partial_prod2, (mean-target).unsqueeze(-1)).squeeze(-1).squeeze(-1)/2
        return - logprob.mean()
    
   
    
    def get_parameters(self, history_input):
        #INPUT: batchsize x 1 x k history x nmodes x 3 
        #TARGET: batchsize x 1 x nmodes x 3 
        #self.history_input = history_input
        #self.history_input = history_input.squeeze(1).squeeze(2)
        self.rot_history = self.Rotate(history_input)
        out = self.encoder(self.rot_history.view( self.rot_history.size(0),-1))
        out_drop = self.dropout(out)
        rot_mean, logvar, _ = self.encode(out_drop)
        
        #rot_history = self.Rotate(history_input )
        #rot_mean, logvar, _ = self.encode(rot_history)
        return rot_mean, logvar
    
    def RotationMatrix(self):
        channels = self.history_input.size(2)
        rotation_matrix = torch.zeros(self.history_input.size(0), channels, channels).to(self.history_input)
        v_t1, norm_t1 = self.history_input[:,-1,:],  self.history_input[:,-1,:].norm(dim=1) 
        v_t2 = self.history_input[:,-2,:]
        v_t3 = self.history_input[:,-3,:] 
        orthonormal_b1 = v_t1/norm_t1.unsqueeze(-1).expand(-1,channels)
        aux_b2 = v_t2 - torch.bmm(v_t2.unsqueeze(1), orthonormal_b1.unsqueeze(-1)).squeeze(-1)*orthonormal_b1 
        orthonormal_b2 = aux_b2/aux_b2.norm(dim=1).unsqueeze(-1).expand(-1,channels)
        rotation_matrix[:,:,0] = orthonormal_b1
        rotation_matrix[:,:,1] = orthonormal_b2
        orthonormal_b3 = torch.linalg.cross(orthonormal_b1, orthonormal_b2)
        sign_flip = torch.bmm(v_t3.unsqueeze(1), orthonormal_b3.unsqueeze(-1)) > 0 
        mask = 2*sign_flip -1
        orthonormal_b3 = mask.squeeze(-1)*orthonormal_b3
        rotation_matrix[:,:,2] = orthonormal_b3
        return rotation_matrix
    
    def Rotate(self, history_input):
        self.history_input = history_input 
         #INPUT: batchsize x k history x nmodes x 3dims
        self.rotation_matrix = self.RotationMatrix()
        #assert (torch.diagonal(torch.bmm(rotation_matrix, rotation_matrix.transpose(1,2)),dim1=1,dim2=2).sum() == float(self.history_input.size(0)*self.history_input.size(-1))).item()
        rot_traj = torch.zeros_like(self.history_input).to(self.history_input)
        for k in range(self.history_input.size(1)):
            rot_traj[:,k,:] = torch.bmm(self.rotation_matrix.transpose(1,2),self.history_input[:,k,:].unsqueeze(-1)).squeeze(-1)
        return rot_traj
    
    def __getitem__(self):
        return self.sigma

    def __getitem__(self):
        return self.rot_history
        
            