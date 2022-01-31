import torch
from systems import HarmonicOscillator

x = torch.tensor([[1.]], requires_grad=True)
order = [2, 2]
dim = 2
mass = 1
omega = 1
domains = [[-0.5, 0.5], [-0.5, 0.5]]
oscillator = HarmonicOscillator(order, dim, mass, omega, domains)
y = oscillator.wavefunction(x)
print(y)
print(torch.autograd.grad(y, x))