import torch
from .dataset import CoordinateData

class MetropolisSampler:
    '''
    Class to implement the Metropolis sampler

    Input
    -----
        system: BaseSystem
            the quantum system to be evaluated
        N: int
            number of coordinates to sample
    '''
    def __init__(self, system, N: int) -> None:
        self.wavefunction = system.wavefunction
        self.domains = system.domains
        self.dim = system.dim
        self.N = N
    
    def prob(self, x):
        # Probability density for the sampler
        return (self.wavefunction(x)) ** 2
    
    def sample(self):
        x0 = torch.ones(1, 1, self.dim)
        for i, (d_min, d_max) in enumerate(self.domains):
            x0[:, :, i] = (d_max - d_min)/2
        
        samples = torch.zeros(self.N, 1, self.dim)
        for i in range(self.N):
            dx = torch.FloatTensor(1, 1, self.dim).uniform_(-0.6, 0.6) # Why 0.6? Who knows.
            x1 = x0 + dx
            if self.prob(x0) < self.prob(x1):
                x0 = x1
            elif torch.rand(1).item() < (self.prob(x1) / self.prob(x0)).item():
                x0 = x1
            samples[i, :, :] = x0[0, 0, :]
        
        return CoordinateData(data=samples, N=self.N)

class RandomSampler:
    def __init__(self, system, N: int) -> None:
        self.wavefunction = system.wavefunction
        self.domains = system.domains
        self.dim = system.dim
        self.N = N
    
    def prob(self, x):
        # Probability density for the sampler
        return (self.wavefunction(x)) ** 2
    
    def sample(self):
        samples = torch.zeros(self.N, 1, self.dim)
        for i, (d_min, d_max) in enumerate(self.domains):
            x = (d_max - d_min) * torch.rand(self.N, 1) + d_min
            samples[:, :, i] = x
        
        return CoordinateData(data=samples, N=self.N)
