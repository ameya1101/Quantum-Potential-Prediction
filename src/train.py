from typing import List
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from systems import HarmonicOscillator, BaseSystem
from utils import MetropolisSampler
import argparse

parser = argparse.ArgumentParser(description='The Metropolis Potential Neural Network.')
parser.add_argument('--system', type=str, default='harmonic', choices=['harmonic'], help='which quantum system to run')
parser.add_argument('--dim', type=int, default=1, choices=range(1, 4), help='coordinate space dimensions')
parser.add_argument('--N', type=int, default=2000, help='number of coordinates to sample')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--mass', type=int, default=1, help='mass term for the harmonic oscillator')
parser.add_argument('--omega', type=int, default=1, help='frequency of the harmonic oscillator')
args = parser.parse_args()

def laplacian(coords, psi):
    xis = [xi.requires_grad_() for xi in coords.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    (dpsi_dxs, ) = torch.autograd.grad(psi, xs_flat, torch.ones_like(psi), create_graph=True)
    laplacian = sum(
        torch.autograd.grad(
            dpsi_dxi, xi, torch.ones_like(dpsi_dxi), retain_graph=True
        )[0]
        for xi, dpsi_dxi in zip(xis, (dpsi_dxs[..., i] for i in range(len(xis))))
    )

    return laplacian # (1, N) tensor


def energy(system: BaseSystem, coords: torch.Tensor, E_pot: torch.Tensor) -> List:
    
    psi = system.wavefunction(coords)
    Hpsi = (-0.5 * laplacian(coords, psi)) + (E_pot.squeeze() * psi.squeeze()) # Laplacian is (1, N), while psi and E_pot are (N, 1, 1). 
                                                                               # squeeze() brings everything to (1, N)
    E_tot = torch.mean(Hpsi / psi.squeeze())

    return E_tot, psi, Hpsi

def train(model, system: BaseSystem, num_epochs, dataloader, optimizer, criterion):
    r0 = torch.ones(1, 1, system.dim) # initial fixed point
    for epoch in range(num_epochs):
        for n_batch, batch in enumerate(dataloader):
            # Convert batch to a variable
            coords = torch.autograd.Variable(batch, requires_grad=True)
            E_pot = model(coords)
            # Find total energy, value of wavefunction and H*psi
            E_tot, psi, Hpsi = energy(system, coords, E_pot)
            # Find potential at initial condition
            E_pot0 = model(r0)
            # backward pass
            optimizer.zero_grad()
            loss = criterion(Hpsi, E_tot * psi.squeeze()) + (E_pot0.squeeze() - system.potential(r0)) ** 2
            loss.backward(retain_graph=True)
            optimizer.step()

            #TODO: Logging metrics