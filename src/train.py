import torch
from torch.optim import Adam
from torch.nn import MSELoss
from systems import HarmonicOscillator
from utils import MetropolisSampler
import argparse

parser = argparse.ArgumentParser(description='The Metropolis Potential Neural Network.')
parser.add_argument('--system', type=str, choices=['harmonic oscillator'], help='which quantum system to rum', required=True)
parser.add_argument('--dim', type=int, default=1, choices=range(1, 4), help='coordinate space dimensions', required=True)
parser.add_argument('--N', type=int, default=2000, help='number of coordinates to sample', required=True)
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs for training', required=True)
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training', required=True)
parser.add_argument('--mass', type=int, default=1, help='mass term for the harmonic oscillator')
parser.add_argument('--omega', type=int, default=1, help='frequency of the harmonic oscillator')
args = parser.parse_args()

def train(model, num_epochs, loader, optimizer):
    for n_batch, batch in enumerate(loader):
        n_data = torch.tensor(n_data, requires_grad=True)
