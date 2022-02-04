import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from models import MPNN
from systems import HarmonicOscillator, BaseSystem
from utils import MetropolisSampler, energy

import argparse
from typing import Tuple

parser = argparse.ArgumentParser(description='The Metropolis Potential Neural Network.')
parser.add_argument('--system', type=str, default='harmonic', choices=['harmonic'], help='which quantum system to run')
parser.add_argument('--dim', type=int, default=1, choices=range(1, 4), help='coordinate space dimensions')
parser.add_argument('--N', type=int, default=2000, help='number of coordinates to sample')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--mass', type=int, default=1, help='mass term for the harmonic oscillator')
parser.add_argument('--omega', type=int, default=1, help='frequency of the harmonic oscillator')
args = parser.parse_args()

def train(
    model: torch.nn.Module, system: BaseSystem, n_epochs: int, 
    dataloader: DataLoader, optimizer, criterion
) -> Tuple:
    
    r0 = torch.ones(1, 1, system.dim) # initial fixed point

    metrics = {
        'loss' : [],
        'energy' : []
    }

    for epoch in range(n_epochs):
        avg_loss = 0
        avg_energy = 0
        for batch in dataloader:
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

            # Log metrics
            avg_loss += loss.item()
            avg_energy += E_tot.item()
        
        metrics['loss'].append(avg_loss / len(dataloader))
        metrics['energy'].append(avg_energy / len(dataloader))

        print(f'Epoch {epoch}: Loss: {avg_loss / len(dataloader)} \t E_tot: {avg_energy / len(dataloader)}')

    return model, metrics

order = 1
dim = args.dim
mass = args.mass
omega = args.omega
domains = [[-5, 5], [-5, 5]]
oscillator = HarmonicOscillator(order=order, dim=dim, mass=mass, omega=omega, domains=domains)

batch_size = args.batch_size
N = args.N
sampler = MetropolisSampler(system=oscillator, N=N)
data = sampler.sample()
dataloader = DataLoader(data, batch_size=batch_size)

model = MPNN(
    in_size=dim,
    out_scaling=12.5,
    delta=0.01
)
criterion = MSELoss()
optimizer = Adam(params=model.parameters())
n_epochs=args.n_epochs

model, metrics = train(
    model=model, system=oscillator, n_epochs=n_epochs,
    dataloader=dataloader, optimizer=optimizer, criterion=criterion
)
