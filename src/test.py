from tracemalloc import start
import torch
from systems import HarmonicOscillator
from models import MPNN
from utils import MetropolisSampler

order = [1, 1]
dim = 2
mass = 1
omega = 1
domains = [[-0.5, 0.5], [-0.5, 0.5]]
oscillator = HarmonicOscillator(order, dim, mass, omega, domains)

coords = torch.tensor([
        [[1., 2.]], 
        [[2., 3.]],
        [[3., 1.]],
], requires_grad=True)

model = MPNN(oscillator.dim, out_scaling=12.5)



# (grad_psi, ) = torch.autograd.grad(psi, coords, torch.ones_like(psi), create_graph=True)
# print(grad_psi)




