import os
import torch 
import numpy as np 
import torch 
import torch.nn as nn


def makedirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

#help function 
def get_default_device():
    "pick GPU if available, else, CPU"
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    "Move data tensor to chosen device"
    if isinstance(data, (list,tuple)): #boolean check type of data at any level 
        return [to_device(x,device) for x in data]
    return data.to(device, non_blocking=True)

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device




def rouse_matrix3D(n_at, n_red):

    return np.array([np.array([np.cos(((p*np.pi)/(n_at))*(n + 0.5))*np.eye(3) 
                for p in range(n_red)]).reshape(3*n_red,3).T 
                for n in range(n_at)]).reshape(3*n_at,3*n_red)

def rouse_matrix1D(n_at, n_red):

    return np.array([np.array([np.cos(((p*np.pi)/(n_at))*(n + 0.5))*np.eye(1) 
                for p in range(n_red)]).reshape(n_red,1).T 
                for n in range(n_at)]).reshape(n_at,n_red)




def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)
    d = torch.distributions.normal.Normal(
        torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device)
    )
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def activation_func(activation):
    return  nn.ModuleDict([
        ['tanh', nn.Tanh()],
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.2, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()],
        ['sigmoid', nn.Sigmoid()]
    ])[activation]





def histogram(data, bins, density=True):
    count = []
    x = np.linspace(np.min(data),np.max(data), bins)
    dx = x[1] - x[0]
    count = [] 
    for i in range(len(x)-1):
        a = data > x[i]
        b = data <= x[i+1]
        count.append(np.logical_and(a,b).sum())
    if density == True:
        norm = sum(count)
        hist = (count)/(norm*dx)
    else:
        hist = count 
    return x,hist, norm
    
def getcorr(data, norm=True):
    nsamples = data.size(0)
    nsteps = data.size(1)
    #data = data - data.mean()
    tau = [t for t in range(0,nsteps)]
    corr = torch.zeros(len(tau)).to(data)
    var = data.var()
    for t in tau:
        diff_tau = torch.bmm(data[:,t:].unsqueeze(1),data[:,:nsteps-t].unsqueeze(-1)).squeeze(-1)/(nsteps-t)
        if norm:
            corr[t] = diff_tau.contiguous().view(-1).mean()/var
        else:
            corr[t] = diff_tau.contiguous().view(-1).mean()
    return corr, tau


def weight_kld(alpha, k, peak_position, epochs):
    steps = np.arange(0,peak_position,1)
    kld_weight = 1/(1 + np.exp(-alpha*(steps-k))) - 1/(1+np.exp(alpha*k))
    for _ in range(epochs//k -2):
        kld_weight = np.concatenate((kld_weight,kld_weight), axis=0)
    return kld_weight


def marginal_prob_std(time, sigma):
    """
    Compute the mean and standard deviation of the conditional distribution p(x(t)|x(0))

    Args:
        t : transformation time 
        sigma: sigma SDE parameter
    """
    t = time.clone().detach().to(time)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma)) 


def diffusion_coeff(time, sigma):
    """Compuation of the SDE diffusion coefficient 

    Args:
        t: A vector of time steps 
        sigma: The sigma SDE parameter
    Returns:
        The vector of diffusion coefficients.
    """
    return (sigma*time.clone().detach()).to(time)


