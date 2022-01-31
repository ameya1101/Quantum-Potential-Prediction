from math import factorial, sqrt
import numpy as np
import torch
from utils import Hermite
from .base import BaseSystem

class HarmonicOscillator(BaseSystem):
    def __init__(self, order, dim, mass, omega, domains) -> None:
        super().__init__(dim, domains)
        assert len(order) == dim, f"Insufficient quantum numbers ({len(order)}) for oscillator with dimension {dim}"
        self._dim = dim
        self._order = order
        self._mass = mass
        self._omega = omega
        self._domains = domains
        self._hbar = 1. # Use natural units

    @staticmethod
    def hermite(x, degree):
        return Hermite.apply(x, degree)
    
    def wavefunction(self, x : torch.Tensor):
        assert x.size()[0] == self._dim, f"Input coordinate dimensions ({x.size()[0]}) do not match system dimensions ({self._dim})."
        
        norm = ((self._mass * self._omega) / (np.pi * self._hbar)) ** 0.25
        psi = torch.tensor(1., dtype=torch.float32, requires_grad=True)
        for i in range(self._dim):
            term1 = (factorial(self._order[i])) * (2 ** self._order[i])
            term2 = (self.hermite(x[i, :], self._order[i]) / sqrt(term1))
            expterms = (-1.0 * self._mass * self._omega * torch.square(x[i, :])) / (2 * self._hbar)
            psi = psi * norm * term2 * torch.exp(expterms)
        return psi
