from math import factorial, sqrt
import numpy as np
import torch
from utils import Hermite
from .base import BaseSystem

class HarmonicOscillator(BaseSystem):
    def __init__(self, order, dim, mass, omega, domains) -> None:
        super().__init__(dim, domains)
        assert len(order) == dim, f"Insufficient quantum numbers ({len(order)}) for oscillator with dimension {dim}"
        self.dim = dim
        self.order = order
        self.mass = mass
        self.omega = omega
        self.domains = domains
        self.hbar = 1. # Use natural units
    
    def potential(self, x):
        return torch.sum(0.5 * self.mass * self.omega ** 2 * x ** 2)

    @staticmethod
    def hermite(x, degree) -> torch.Tensor:
        return Hermite.apply(x, degree)
    
    def wavefunction(self, x : torch.Tensor) -> torch.Tensor:
        assert x.size()[2] == self.dim, f"Input coordinate dimensions ({x.size()[1]}) do not match system dimensions ({self.dim})."
        
        norm = ((self.mass * self.omega) / (np.pi * self.hbar)) ** 0.25
        psi = torch.ones(1, 1, 1, requires_grad=True)

        for i in range(self.dim):
            term1 = (factorial(self.order[i])) * (2 ** self.order[i])
            term2 = (self.hermite(x[:, :, i].unsqueeze(1), self.order[i]) / sqrt(term1)) # (N, 1) -> (N, 1, 1)
            expterms = (-1.0 * self.mass * self.omega * torch.square(x[:, :, i].unsqueeze(1))) / (2 * self.hbar) # (N, 1) -> (N, 1, 1)
            psi = psi * (norm * term2 * torch.exp(expterms))
        return psi