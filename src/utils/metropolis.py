import torch

class MetropolisSampler:
    def __init__(self, wavefunction, N: int) -> None:
        self.wavefunction = wavefunction
        self.domains = wavefunction._domains
        self.dim = wavefunction._dim
        self.N = N
        self.samples = torch.zeros(self.dim, self.N)
    
    def prob(self, x):
        return (self.wavefunction(x)) ** 2
    
    def sample(self):
        x0 = torch.ones_like(self.dim, 1) * 0.5
        samples = torch.zeros(self.dim, self.N)
        for i in range(self.N):
            dx = torch.FloatTensor(self.dim, 1).uniform_(-0.6, 0.6)
            x1 = x0 + dx
            if self.prob(x0) < self.prob(x1):
                x0 = x1
            elif torch.random < (self.prob(x1) / self.prob(x0)):
                x0 = x1
            
            samples[:, i] = x0[:, 0]

