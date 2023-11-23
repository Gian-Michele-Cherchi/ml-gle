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


def msd_bf(x, tau):
    diff2 = (x[:,:x.shape[1]-tau] - x[:,tau:])**2
    return diff2.mean(dim=1).mean()

#ACF FFT Algorithmv. Cost O(nlog(n))
def autocorrFFT(x):
  N=x.size(1)
  batchsize = x.size(0)
  F = torch.fft.fft(x, n=2*N)
  PSD = F*F.conj()
  res = torch.fft.ifft(PSD)
  res= (res[:,:N]).real  
  n=N*torch.ones(N)-torch.arange(0,N)
  return res/n.unsqueeze(0).expand(batchsize,-1).to(x)

 # MSD FFT Algorithm
def msd_fft(r):
  N=r.size(1)
  D=torch.square(r).to(r) 
  #torch.cuda.empty_cache()
  D=torch.cat([D, torch.zeros(D.size(0),1).to(r)], dim=1)
  S2=autocorrFFT(r)
  Q=2*D.sum(dim=1)
  S1=torch.zeros(r.size(0),N).to(r)
  for m in range(N):
      Q=Q-D[:,m-1]-D[:,N-m]
      S1[:,m]=Q/(N-m)
  return S1-2*S2

    
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






