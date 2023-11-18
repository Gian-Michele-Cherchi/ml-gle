import os
import yaml
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
PROJECTNAME = config["PROJECTNAME"]
USERPATH = config["USERPATH"]
SYSTEM = config["SYSTEM"]
FULLPATH  = os.path.join(USERPATH, PROJECTNAME)
SUBMODULE = os.path.join(FULLPATH, "ia")
import sys 
sys.path.append(SUBMODULE)
import torch.nn as nn 
from library.utils import * 
from library.layers import *
from library.score_net import *
from .loss_functs import *

class DyamicalLinear(nn.Module):
    
    def __init__(self, 
                input_length: int,
                in_channels: int,
                lat_dims: int=16,
                pred_var_dims: int=1,
                activation: str='relu',
                ):
        super(DyamicalLinear, self).__init__()
        self.in_channels = in_channels
        self.input_feat = in_channels*input_length 
        self.pred_var_dims = pred_var_dims
        self.lat_dims = lat_dims
       
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_feat, out_features=self.input_feat),
            activation_func(activation),
            nn.Linear(in_features=self.input_feat, out_features=self.input_feat//2),  
            activation_func(activation),
            nn.Linear(in_features=self.input_feat//2, out_features=self.input_feat//4),
            activation_func(activation),
            nn.Linear(in_features=self.input_feat//4, out_features=self.input_feat//8),
            activation_func(activation),
            #nn.Linear(in_features=self.input_feat//8, out_features=self.input_feat//16),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//2, out_features=self.input_feat//4),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//4, out_features=self.input_feat//8),
            #activation_func(activation), 
        )
        
        self.enc_to_mean = nn.Linear(self.input_feat//8, self.pred_var_dims)
        self.enc_to_var = nn.Linear(self.input_feat//8, self.pred_var_dims)
       
            
    def forward(self, history_input, target_acc=None):
        #INPUT: batchsize x 1 x k history x nmodes
        #TARGET: batchsize x 1 x nmodes
        self.history_input = history_input.squeeze(1)
        #ref = self.history_input[:,-1,:].unsqueeze(1)
        
        mean1, logvar1 = self.encode(self.history_input)
        mean2, logvar2 = self.encode(-self.history_input)
        mean = (mean1-mean2)/2
        logvar = torch.log((torch.exp(logvar1)+torch.exp(logvar2))/2)
        if target_acc is None:
            return mean, logvar
        else:
            loss = self.NLL(mean, logvar, target_acc)
            return loss, [mean,logvar]
               

    def encode(self, history_input): 
        out = self.encoder(history_input.view( history_input.size(0),-1))
        means = self.enc_to_mean(out)
        logvarsquared = self.enc_to_var(out)
        return means, logvarsquared
    
    def NLL(self, mean,logvar, target):
        logprob = -(1/2)*logvar.sum(dim=1) - torch.bmm(torch.exp(-logvar).unsqueeze(-1), ((mean.unsqueeze(-1)-target)**2)).squeeze(-1).squeeze(-1)/2
        return - logprob.mean(dim=0)


    
    

class DyamicalLinear3D(nn.Module):
    
    def __init__(self, 
                input_length: int,
                in_channels: int,
                hidden_channels: int=1, 
                lat_dims: int=16,
                pred_var_dims: int=1,
                activation: str='relu',
                gyration_ref: bool=False,
                diagonal: bool=False
                ):
        super(DyamicalLinear3D, self).__init__()
        self.gyration_ref = gyration_ref
        self.in_channels = in_channels
        self.input_feat = 3*in_channels*input_length 
        self.hidden_dim = hidden_channels
        self.pred_var_dims = pred_var_dims
        self.lat_dims = lat_dims
        self.diagonal = diagonal
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_feat, out_features=self.input_feat//2),
            activation_func(activation),
            nn.Linear(in_features=self.input_feat//2, out_features=self.input_feat//4),  
            activation_func(activation),
            #nn.Linear(in_features=self.input_feat//4, out_features=self.input_feat//8),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//8, out_features=self.input_feat//16),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//16, out_features=self.input_feat//32),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//32, out_features=self.input_feat//64),   
        )
        
        self.enc_to_mean = nn.Linear(self.input_feat//4, self.pred_var_dims)
        if diagonal:
            self.enc_to_var = nn.Linear(self.input_feat//4, self.pred_var_dims)
        else:
            self.alpha = int(self.pred_var_dims*(1+ (self.pred_var_dims-1)/2)) # n*(n-1)/2
            self.enc_to_var = nn.Linear(self.input_feat//4, self.alpha)
            
    def forward(self, history_input, target_acc=None) -> list:
        #INPUT: batchsize x 1 x k history x nmodes x3
        #TARGET: batchsize x 1 x nmodes x 3
        #ref = history_input
        self.history_input = history_input.squeeze(1)
        self.rot_history = self.Rotate()
        rot_mean, logvar = self.encode(self.rot_history)
        if target_acc is None:
            return rot_mean, logvar
        else:
            if self.gyration_ref:
                rot_target = torch.bmm(
                    target_acc.squeeze(1), 
                    self.rotation_matrix)
                n_oddmodes = rot_target[:,1::2,:].size(1)
                rot_target[:,1::2,:]= torch.mul(self.odd_symmetry.unsqueeze(-1).expand(-1,n_oddmodes,3),
                                             rot_target[:,1::2,:])
                if self.diagonal:
                    loss = self.NLL(rot_mean, logvar, rot_target.view(-1, 3*self.in_channels), self.diagonal)
                else:
                    loss = self.NLL(rot_mean, logvar, rot_target.view(-1, 3*self.in_channels))
                return loss, [rot_mean,logvar]
            else:
                rot_target = torch.bmm(self.rotation_matrix.transpose(1,2),target_acc.swapaxes(1,2)).squeeze(-1) #target rotation to frame (t-1,t-2,t-3)
                if self.diagonal:
                    loss = self.NLL(rot_mean.squeeze(-1), logvar, rot_target, self.diagonal)
                else:
                    loss = self.NLL(rot_mean.squeeze(-1), logvar, rot_target)
                return loss, [rot_mean,logvar]

    def encode(self, history_input): 
        out = self.encoder(history_input.view( history_input.size(0),-1))
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
            return means, [diag, lower]
    
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
    
    def GyrationTensorDecomposition(self):
            gyr_matrix= torch.bmm(self.history_input[:,-1,1:,:].swapaxes(1,2), self.history_input[:,-1,1:,:])/((self.history_input.size(2)-1))
            return torch.linalg.eigh(gyr_matrix)
            
    
    def Rotate(self):
         #INPUT: batchsize x k history x nmodes x 3dims
        if  not self.gyration_ref:
            self.rotation_matrix = self.RotationMatrix()
            #assert (torch.diagonal(torch.bmm(rotation_matrix, rotation_matrix.transpose(1,2)),dim1=1,dim2=2).sum() == float(self.history_input.size(0)*self.history_input.size(-1))).item()
            rot_traj = torch.zeros_like(self.history_input).to(self.history_input)
            for k in range(self.history_input.size(1)):
                rot_traj[:,k,:] = torch.bmm(self.rotation_matrix.transpose(1,2),self.history_input[:,k,:].unsqueeze(-1)).squeeze(-1)
            return rot_traj
        else:
            _, self.rotation_matrix = self.GyrationTensorDecomposition()
            #self.rotation_matrix = self.rotation_matrix.real
            # batchsize x 1 x 3 
            check_dir = torch.bmm(self.history_input[:,-1,:1,:], self.rotation_matrix) >= 0
            flip_eigenvector = 2*check_dir -1
            self.rotation_matrix[:,:,0] = torch.mul(flip_eigenvector[:,:1,0], self.rotation_matrix[:,:,0])
            self.rotation_matrix[:,:,1] = torch.mul(flip_eigenvector[:,:1,1], self.rotation_matrix[:,:,1])
            self.rotation_matrix[:,:,2] = torch.mul(flip_eigenvector[:,:1,2], self.rotation_matrix[:,:,2])
            rot_traj = torch.zeros_like(self.history_input).to(self.history_input)
            for k in range(self.history_input.size(1)):
                rot_traj[:,k,] = torch.bmm(self.history_input[:,k,:,:], self.rotation_matrix)
            check  = self.history_input[:,-1,1:2,0] >= 0 
            self.odd_symmetry = 2*check -1 
            n_oddmodes = rot_traj[:,:,1::2,: ].size(2)
            #self.odd_symmetry = self.odd_symmetry.unsqueeze(1).unsqueeze(-1).expand(-1,self.history_input.size(1),n_oddmodes,3)
            rot_traj[:,:,1::2,: ]= torch.mul(self.odd_symmetry.unsqueeze(1).unsqueeze(-1).expand(-1,self.history_input.size(1),n_oddmodes,3),
                                             rot_traj[:,:,1::2,:])
            
            return rot_traj
        
    
    def NLL(self, mean,logvar, target, diagonal=False):
        diag, lower = logvar
        partial_prod1 = torch.bmm(lower, diag)
        sigma = torch.bmm(partial_prod1, lower.transpose(1,2))
        sigma_inverse = torch.inverse(self.sigma)
        partial_prod2  = torch.bmm((mean-target).unsqueeze(1), sigma_inverse)
        logprob = -(1/2)*torch.diagonal(diag, dim1=1, dim2=2).sum(dim=1) - torch.bmm(partial_prod2, (mean-target).unsqueeze(-1)).squeeze(-1).squeeze(-1)/2
        return - logprob.mean(dim=0)
    
    def __getitem__(self):
        return self.sigma

    def __getitem__(self):
        return self.rot_history   
        


class DyamicalConv3D(nn.Module):
    
    
    
    def __init__(self, 
                input_length: int,
                in_channels: int,
                hidden_channels: int=1, 
                lat_dims: int=16,
                pred_var_dims: int=1,
                activation: str='relu',
                full_cov: bool=False,
                gyration_ref: bool=False
                ):
        super(DyamicalConv3D, self).__init__()
        self.gyration_ref = gyration_ref
        self.input_feat = input_length # input dimensions is concatenation of av. velocity history + rouse mode history + (maybe) infos
        self.hidden_dim = hidden_channels
        self.pred_var_dims = pred_var_dims
        self.lat_dims = lat_dims
        self.corr = full_cov
        self.encoder =  nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_channels*2, kernel=15, stride=1, activation=activation, dilation=1),
            #nn.MaxPool1d(2),
            ConvBlock(in_channels=in_channels*2, out_channels=in_channels*2, kernel=1, stride=1, activation=activation,dilation=1),
            #nn.MaxPool1d(2),
            #ConvBlock(in_channels=in_channels*2, out_channels=in_channels*4, kernel=3, stride=1, activation=activation, dilation=1),
            #nn.MaxPool1d(2),
            #ConvBlock(in_channels=in_channels*8, out_channels=in_channels*16, kernel=1, stride=1, activation=activation, dilation=1),
            #nn.MaxPool1d(2),
            #ConvBlock(in_channels=in_channels*32, out_channels=in_channels*64, kernel=1, stride=1, activation=activation),
        )
        # l_out = l_in -d*(k-1)
        self.enc_to_latspace = nn.Linear(18*2*in_channels, self.pred_var_dims*self.lat_dims)
        self.enc_to_mean = nn.Linear(self.pred_var_dims*self.lat_dims, self.pred_var_dims)
        
        self.alpha = int(self.pred_var_dims*(1+ (self.pred_var_dims-1)/2))
        self.enc_to_var = nn.Linear(self.pred_var_dims*self.lat_dims, self.alpha)
      

    def forward(self, history_input, target_acc=None):
        #INPUT: batchsize x 1 x k history x nmodes x3
        #TARGET: batchsize x 1 x nmodes x 3
        self.history_input = history_input.squeeze(1)
        self.rot_history = self.Rotate()
        rot_mean, logvar = self.encode(self.rot_history.swapaxes(1,2))
        if target_acc is None:
            return rot_mean, logvar
        else:
            rot_target = torch.bmm(self.rotation_matrix.transpose(1,2),target_acc.swapaxes(1,2)).squeeze(-1) #target rotation to frame (t-1,t-2,t-3)
            loss = self.NLL(rot_mean.squeeze(-1), logvar, rot_target)
            return loss, [rot_mean,logvar]
        

    def encode(self, history_input): 
        outconv = self.encoder(history_input)
        flatten = torch.flatten(outconv, start_dim=1, end_dim=-1)
        flatten = self.enc_to_latspace(flatten)
        means = self.enc_to_mean(flatten)
        elements = self.enc_to_var(flatten)
        lower = torch.zeros(elements.size(0), self.pred_var_dims, self.pred_var_dims).to(elements)
        batched_eye = torch.eye(self.pred_var_dims).unsqueeze(0).expand(elements.size(0),-1,-1).to(elements)
        lower = lower + batched_eye
        index = np.cumsum([n for n in range(0,self.pred_var_dims)])
        for row in range(1,self.pred_var_dims):
            lower[:,row,:row] = elements[:,index[row-1]:index[row]]

        diag = torch.exp(elements[:,self.alpha-self.pred_var_dims:]).unsqueeze(1)*batched_eye
        partial_prod1 = torch.bmm(lower, diag)
        self.sigma = torch.bmm(partial_prod1, lower.transpose(1,2))
        return means, [diag, lower]
    
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
    
    def GyrationTensorDecomposition(self):
            gyr_matrix= torch.bmm(self.history_input[:,-1,1:,:].swapaxes(1,2), self.history_input[:,-1,1:,:])/(self.history_input.size(2)-1)
            return torch.linalg.eig(gyr_matrix)
            
    
    def Rotate(self):
        if  not self.gyration_ref:
            self.rotation_matrix = self.RotationMatrix()
            #assert (torch.diagonal(torch.bmm(rotation_matrix, rotation_matrix.transpose(1,2)),dim1=1,dim2=2).sum() == float(self.history_input.size(0)*self.history_input.size(-1))).item()
            rot_traj = torch.zeros_like(self.history_input).to(self.history_input)
            for k in range(self.history_input.size(1)):
                rot_traj[:,k,:] = torch.bmm(self.rotation_matrix.transpose(1,2),self.history_input[:,k,:].unsqueeze(-1)).squeeze(-1)
            return rot_traj
        else:
            _ ,self.rotation_matrix = self.GyrationTensorDecomposition()
            rot_traj = torch.zeros_like(self.history_input).to(self.history_input)
            #for k in range(self.history_input.size(1)):
            #    rot_traj[:,k,:] = torch.bmm(
            #return rot_traj
        
    
    def NLL(self, mean,logvar, target):
        diag, lower = logvar
        partial_prod1 = torch.bmm(lower, diag)
        sigma = torch.bmm(partial_prod1, lower.transpose(1,2))
        sigma_inverse = torch.inverse(self.sigma)
        partial_prod2  = torch.bmm((mean-target).unsqueeze(1), sigma_inverse)
        logprob = -(1/2)*torch.diagonal(diag, dim1=1, dim2=2).sum(dim=1) - torch.bmm(partial_prod2, (mean-target).unsqueeze(-1)).squeeze(-1).squeeze(-1)/2
        return - logprob.mean(dim=0)
    
    def __getitem__(self):
        return self.sigma

    def __getitem__(self):
        return self.rot_history  
    
    
   
    
    
class RecurrentDynamical3D(nn.Module):
    
    def __init__(self, 
                input_length: int,
                in_channels: int,
                hidden_dims: int=1, 
                lat_dims: int=16,
                pred_var_dims: int=1,
                activation: str='relu',
                gyration_ref: bool=False
                ):
        super(RecurrentDynamical3D, self).__init__()
        self.gyration_ref = gyration_ref
        self.in_channels = in_channels
        self.input_features = 3*in_channels #3 * n_modes
        self.hidden_dims = hidden_dims # hidden state dimensions 
        self.pred_var_dims = pred_var_dims
        self.lat_dims = lat_dims
        self.input_length = input_length
        
        # INPUT SIZE: batchsize x input_length x 3*in_channels 
        # OUTPUT: batchsize x input_length x D*hidden_dims
        # CELL STATE: (D*num_layers, batchsize, hidden_dims)
        self.recurrent_block = nn.LSTM(self.input_features, self.hidden_dims, batch_first=True) #Memory Block (GRU, LSTM)
        
        #self.fc = self.linear(self.hidden_dim, self.lat_dims) #features dim reduction 
        
        self.enc_to_mean = nn.Linear(self.hidden_dims*self.input_length, self.pred_var_dims)
        self.alpha = int(self.pred_var_dims*(1+ (self.pred_var_dims-1)/2)) # n*(n-1)/2
        self.enc_to_var = nn.Linear(self.hidden_dims*self.input_length, self.alpha)

    def forward(self, history_input, hidden_states,target_acc=None):
        #INPUT: batchsize x 1 x input_length x nmodes x3
        #TARGET: batchsize x 1 x nmodes x 3
        self.history_input = history_input.squeeze(1)
        self.rot_history = self.Rotate()
        rot_mean, logvar, hidden_states = self.encode(self.rot_history, hidden_states)
        if target_acc is None:
            return rot_mean, logvar, hidden_states
        else:
            if self.gyration_ref:
                rot_target = torch.bmm(
                    target_acc.squeeze(1), 
                    self.rotation_matrix)
                n_oddmodes = rot_target[:,1::2,:].size(1)
                rot_target[:,1::2,:]= torch.mul(self.odd_symmetry.unsqueeze(-1).expand(-1,n_oddmodes,3),
                                             rot_target[:,1::2,:])
                loss = self.NLL(rot_mean, logvar, rot_target.view(-1, 3*self.in_channels))
                return loss, [rot_mean,logvar], hidden_states
            else:
                rot_target = torch.bmm(self.rotation_matrix.transpose(1,2),target_acc.swapaxes(1,2)).squeeze(-1) #target rotation to frame (t-1,t-2,t-3)
                loss = self.NLL(rot_mean.squeeze(-1), logvar, rot_target)
                return loss, [rot_mean,logvar], hidden_states

    def encode(self, history_input, hidden_states):
        h_t, c_t = hidden_states[0], hidden_states[1] 
        out, hidden_states = self.recurrent_block(history_input.view(history_input.size(0),-1,3*self.in_channels), (h_t,c_t))
        means = self.enc_to_mean(out.contiguous().view(-1, self.input_length*self.hidden_dims))
        elements = self.enc_to_var(out.contiguous().view(-1, self.input_length*self.hidden_dims))
        
        lower = torch.zeros(elements.size(0), self.pred_var_dims, self.pred_var_dims).to(elements)
        batched_eye = torch.eye(self.pred_var_dims).unsqueeze(0).expand(elements.size(0),-1,-1).to(elements)
        lower = lower + batched_eye
        index = np.cumsum([n for n in range(0,self.pred_var_dims)])
        for row in range(1,self.pred_var_dims):
            lower[:,row,:row] = elements[:,index[row-1]:index[row]]

        diag = torch.exp(elements[:,self.alpha-self.pred_var_dims:]).unsqueeze(1)*batched_eye
        partial_prod1 = torch.bmm(lower, diag)
        self.sigma = torch.bmm(partial_prod1, lower.transpose(1,2))
        return means, [diag, lower], hidden_states
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
    
    def GyrationTensorDecomposition(self):
            gyr_matrix= torch.bmm(self.history_input[:,-1,1:,:].swapaxes(1,2), self.history_input[:,-1,1:,:])/((self.history_input.size(2)-1))
            return torch.linalg.eigh(gyr_matrix)
            
    
    def Rotate(self):
         #INPUT: batchsize x k history x nmodes x 3dims
        if  not self.gyration_ref:
            self.rotation_matrix = self.RotationMatrix()
            #assert (torch.diagonal(torch.bmm(rotation_matrix, rotation_matrix.transpose(1,2)),dim1=1,dim2=2).sum() == float(self.history_input.size(0)*self.history_input.size(-1))).item()
            rot_traj = torch.zeros_like(self.history_input).to(self.history_input)
            for k in range(self.history_input.size(1)):
                rot_traj[:,k,:] = torch.bmm(self.rotation_matrix.transpose(1,2),self.history_input[:,k,:].unsqueeze(-1)).squeeze(-1)
            return rot_traj
        else:
            _, self.rotation_matrix = self.GyrationTensorDecomposition()
            #self.rotation_matrix = self.rotation_matrix.real
            # batchsize x 1 x 3 
            check_dir = torch.bmm(self.history_input[:,-1,:1,:], self.rotation_matrix) >= 0
            flip_eigenvector = 2*check_dir -1
            self.rotation_matrix[:,:,0] = torch.mul(flip_eigenvector[:,:1,0], self.rotation_matrix[:,:,0])
            self.rotation_matrix[:,:,1] = torch.mul(flip_eigenvector[:,:1,1], self.rotation_matrix[:,:,1])
            self.rotation_matrix[:,:,2] = torch.mul(flip_eigenvector[:,:1,2], self.rotation_matrix[:,:,2])
            rot_traj = torch.zeros_like(self.history_input).to(self.history_input)
            for k in range(self.history_input.size(1)):
                rot_traj[:,k,] = torch.bmm(self.history_input[:,k,:,:], self.rotation_matrix)
            check  = self.history_input[:,-1,1:2,0] >= 0 
            self.odd_symmetry = 2*check -1 
            n_oddmodes = rot_traj[:,:,1::2,: ].size(2)
            #self.odd_symmetry = self.odd_symmetry.unsqueeze(1).unsqueeze(-1).expand(-1,self.history_input.size(1),n_oddmodes,3)
            rot_traj[:,:,1::2,: ]= torch.mul(self.odd_symmetry.unsqueeze(1).unsqueeze(-1).expand(-1,self.history_input.size(1),n_oddmodes,3),
                                             rot_traj[:,:,1::2,:])
            
            return rot_traj
        
    
    def NLL(self, mean,logvar, target):
        diag, lower = logvar
        partial_prod1 = torch.bmm(lower, diag)
        sigma = torch.bmm(partial_prod1, lower.transpose(1,2))
        sigma_inverse = torch.inverse(self.sigma)
        partial_prod2  = torch.bmm((mean-target).unsqueeze(1), sigma_inverse)
        logprob = -(1/2)*torch.diagonal(diag, dim1=1, dim2=2).sum(dim=1) - torch.bmm(partial_prod2, (mean-target).unsqueeze(-1)).squeeze(-1).squeeze(-1)/2
        return - logprob.mean(dim=0)
    
    def __getitem__(self):
        return self.sigma

    def __getitem__(self):
        return self.rot_history   
    
    
EPS = 1e-5
class DyamicalScoreSDE(nn.Module):
    
    def __init__(self, 
                input_length: int,
                in_channels: int,
                marginal_prob_std_fn, 
                pred_var_dims: int=1,
                activation: str='relu',
                ):
        super(DyamicalScoreSDE, self).__init__()
        self.in_channels = in_channels
        self.input_feat = in_channels*input_length 
        self.pred_var_dims = pred_var_dims
        bias = True
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_feat, out_features=self.input_feat, bias=bias),
            activation_func(activation),
            nn.Linear(in_features=self.input_feat, out_features=self.input_feat//2, bias=bias),  
            activation_func(activation),
            nn.Linear(in_features=self.input_feat//2, out_features=self.input_feat//4, bias=bias),
            activation_func(activation),
            nn.Linear(in_features=self.input_feat//4, out_features=3, bias=bias),
            activation_func(activation),
            #nn.Linear(in_features=self.input_feat//8, out_features=self.input_feat//16),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//2, out_features=self.input_feat//4),
            #activation_func(activation),
            #nn.Linear(in_features=self.input_feat//4, out_features=self.input_feat//8),
            #activation_func(activation), 
        )
        self.marginal_cond_std = marginal_prob_std_fn
        #self.score_net =  ConditionalScoreNet1D(marginal_prob_std=self.marginal_cond_std, cond_input_dim=self.input_feat//2)
        self.score_net = ConditionalScoreNet(
            marginal_prob_std=self.marginal_cond_std,
            cond_length=self.input_feat//8,
            residual_channels=4,
            dilation_cycle=2,
            n_layers=4,
            embed_dim=64
        )
        self.enc_to_mean = nn.Linear(3, self.pred_var_dims)
        self.enc_to_var = nn.Linear(3, self.pred_var_dims)
       
            
    def forward(self, history_input, target_x, target_vel):
        #INPUT: batchsize x 1 x k history x nmodes
        #TARGET: batchsize x 1 x nmodes
        self.history_input = history_input.squeeze(1)
        target_x = target_x.squeeze(-1)
        #ref = self.history_input[:,-1,:].unsqueeze(1)
        
        mean1, logvar1, encoded1 = self.encode(self.history_input)
        mean2, logvar2, encoded2 = self.encode(-self.history_input)
        mean = (mean1-mean2)/2
        logvar = torch.log((torch.exp(logvar1)+torch.exp(logvar2))/2)
        
        # Random Uniform[0,1] evaluation time step for loss evaluation 
        random_t = torch.rand(target_x.size(0), device=target_x.device) * (1-EPS) + EPS 
        z = torch.randn_like(target_x)
        std = self.marginal_cond_std(random_t.clone().detach())
        perturbed_x = target_x + z * std[:, None]
        score1 = self.score_net(perturbed_x, encoded1, random_t)
        score2 = self.score_net(-perturbed_x, encoded2, random_t)
        score = (score1-score2)/2
        # Loss evaluation 
        score_loss = self.fisher_div(score, std, z)
        dyn_loss = self.NLL(mean, logvar, target_vel)
        loss = score_loss + dyn_loss
        return loss, [mean,logvar], [dyn_loss, score_loss]
               
    
    def encode(self, history_input): 
        out = self.encoder(history_input.view( history_input.size(0),-1))
        means = self.enc_to_mean(out)
        logvarsquared = self.enc_to_var(out)
        return means, logvarsquared, out
    
    def NLL(self, mean,logvar, target):
        logprob = -(1/2)*logvar.sum(dim=1) - torch.bmm(torch.exp(-logvar).unsqueeze(-1), ((mean.unsqueeze(-1)-target)**2)).squeeze(-1).squeeze(-1)/2
        return - logprob.mean(dim=0)
    
    def fisher_div(model, score, std, z):
        loss =  torch.mean((score * std[:, None] + z)**2)
        return loss
    
    def get_score(self, history_input, target_x, batch_steps):
        
        self.history_input = history_input.squeeze(1)
        _, _, encoded1 = self.encode(self.history_input)
        _, _, encoded2 = self.encode(-self.history_input)
       
        score1 = self.score_net(target_x, encoded1, batch_steps)
        score2 = self.score_net(-target_x, encoded2, batch_steps)
        score = (score1-score2)/2
        return score
    
    def get_parameters(self, history_input):
        #INPUT: batchsize x 1 x k history x nmodes
        #TARGET: batchsize x 1 x nmodes
        self.history_input = history_input.squeeze(1)
        #target_x = target_x.squeeze(-1)
        mean1, logvar1, _ = self.encode(self.history_input)
        mean2, logvar2, _ = self.encode(-self.history_input)
        mean = (mean1-mean2)/2
        logvar = torch.log((torch.exp(logvar1)+torch.exp(logvar2))/2) #type:ignore
        return mean, logvar
    


class DyamicalScoreSDE_3D(nn.Module):
    
    def __init__(self, 
                input_length: int,
                in_channels: int,
                residual_channels: int, 
                marginal_prob_std_fn, 
                pred_var_dims: int=3,
                activation: str='relu',
                diagonal: bool=False,
                scoreonly: bool=False,
                score_nlayers: int = 4,
                reg: float=0, 
                ):
        super(DyamicalScoreSDE_3D, self).__init__()
        self.in_channels = in_channels
        self.input_feat = 3*in_channels*input_length 
        self.pred_var_dims = pred_var_dims
        self.diagonal = diagonal
        self.res_channels = residual_channels
        self.score_only = scoreonly
        bias = True
        self.reg= reg 
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
        self.marginal_cond_std = marginal_prob_std_fn
        #self.score_net =  ConditionalScoreNet1D(marginal_prob_std=self.marginal_cond_std, cond_input_dim=self.input_feat//2)
        self.score_net = ConditionalScoreNet(
            marginal_prob_std=self.marginal_cond_std,
            cond_length=self.input_feat//2,
            residual_channels=self.res_channels,
            dilation_cycle=2,
            n_layers=score_nlayers,
            embed_dim=64
        )
        
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
        
        # Random Uniform[0,1] evaluation time step for loss evaluation 
        random_t = torch.rand(target_x.size(0), device=target_x.device) * (1-EPS) + EPS 
        z = torch.randn_like(target_x)
        std = self.marginal_cond_std(random_t)
        perturbed_x = rot_target_x + z * std[:, None, None]
        score = self.score_net(perturbed_x, encoded.view(encoded.size(0), self.res_channels, 3), random_t)
        score_loss = self.fisher_div(score, std, z.squeeze(1))
        
        if self.score_only:
            loss = score_loss
            dyn_loss = torch.tensor(0).to(history_input) 
            return loss
        else:
            if int(self.reg) == 0:
                score_loss = torch.tensor(0).to(history_input)
            
            dyn_loss = self.NLL(rot_mean, logvar, rot_target_vel)
            loss = dyn_loss + self.reg*score_loss
            return loss, [rot_mean,logvar], [dyn_loss, score_loss], encoded 
               
    
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
    
    def fisher_div(model, score, std, z):
        loss =  torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
        return loss
    
    
    def get_score(self, rot_history, target_x, batch_steps):
        #self.history_input = history_input.squeeze(2)
        #self.history_input = history_input.squeeze(2)
        #rot_history = self.Rotate()
        _, _, encoded = self.encode(rot_history)
        
        #target_x = target_x.squeeze(-1)
        #rot_target_x = torch.bmm(self.rotation_matrix.transpose(1,2),target_x.swapaxes(1,2)).squeeze(-1)
        score = self.score_net(target_x, encoded.view(encoded.size(0), self.res_channels, 3), batch_steps)
        return score
    
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
        
            