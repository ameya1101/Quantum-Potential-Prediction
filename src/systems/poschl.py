from mimetypes import init
import torch
from .base import BaseSystem


class PoschlTeller(BaseSystem):
    '''
    A class to implement the the Pöschl-Teller potential.

    Currently, only the wavefunction for λ = 2, μ = 1 is supported. 

    Inputs
    ------
        dim: int
            dimensionality of the system
        domains: List[List]
            A list containing upper and lower coordinate bounds for each dimension

    '''
    def __init__(self, dim, domain) -> None:
        super().__init__(dim, domain)

    def wavefunction(self, x):
        return ((torch.abs(torch.tanh(x))) * (1 / torch.cosh(x)))

    def potential(self, x):
        return -3 * (1 / torch.square(torch.cosh(x)))