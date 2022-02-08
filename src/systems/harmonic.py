from math import factorial, sqrt, pi
import torch
from .base import BaseSystem

class HarmonicOscillator(BaseSystem):
    '''
    A class to implement the n-dimensional quantium simple harmonic oscillator.

    Inputs
    ------
        order: List
            quantum numbers describing the system
        dim: int
            dimensionality of the system
        mass: float
            mass of the harmonic oscillator
        omega: float
            natural frequency of the harmonic oscillator
        domains: List[List]
            A list containing upper and lower coordinate bounds for each dimension

    '''
    def __init__(
        self, order, 
        dim, mass, 
        omega, domains
    ) -> None:
        super().__init__(dim, domains)
        assert len(order) == dim, f"Insufficient quantum numbers ({len(order)}) for oscillator with dimension {dim}"
        self.order = order
        self.mass = mass
        self.omega = omega
        self.hbar = 1. # Use natural units
    
    def potential(self, x):
        return torch.sum(0.5 * self.mass * (self.omega ** 2) * (x ** 2))

    def hermite(self, x, degree) -> torch.Tensor:
        # (Physicists') Hermite polynomial using recurrence relation
        if degree == 0:
            return 1
        elif degree == 1:
            return 2 * x
        else:
            return 2 * x * self.hermite(x, degree - 1) - 2 * (degree - 1) * self.hermite(x, degree - 2)
    
    def wavefunction(self, x : torch.Tensor) -> torch.Tensor:
        assert x.size()[2] == self.dim, f"Input coordinate dimensions ({x.size()[1]}) do not match system dimensions ({self.dim})."
        
        norm = ((self.mass * self.omega) / (pi * self.hbar)) ** 0.25
        psi = torch.ones(1, 1, 1, requires_grad=True)

        for i in range(self.dim):
            term1 = (factorial(self.order[i])) * (2 ** self.order[i])
            term2 = (self.hermite(x[:, :, i].unsqueeze(1), self.order[i]) / sqrt(term1)) # (N, 1) -> (N, 1, 1)
            expterms = (-1.0 * self.mass * self.omega * torch.square(x[:, :, i].unsqueeze(1))) / (2 * self.hbar) # (N, 1) -> (N, 1, 1)
            psi = psi * (norm * term2 * torch.exp(expterms))
        return psi
        