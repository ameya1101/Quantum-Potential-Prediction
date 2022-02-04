import torch
from typing import List

def energy(system, coords: torch.Tensor, E_pot: torch.Tensor) -> List:
    '''
    Method to compute the energy of a quantum system. 

    Input
    -----
        system: BaseSystem
            the quantum system to be evaulated
        coords: torch.Tensor
            input coordinates
        E_pot: torch.Tensor
            predicted potential energy of the system
    Returns
    --------
        E_tot: torch.Tensor
            The total energy of the systen
        psi: torch.Tensor
            the wavefunction evaluated at the input coordinates
        Hpsi: torch.Tensor
            result of applying the Hamiltonian on psi
    '''

    xis = [xi.requires_grad_() for xi in coords.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    psi = system.wavefunction(xs_flat.view_as(coords))
    (dpsi_dxs, ) = torch.autograd.grad(psi, xs_flat, torch.ones_like(psi), create_graph=True)
    laplacian = sum(
        torch.autograd.grad(
            dpsi_dxi, xi, torch.ones_like(dpsi_dxi), retain_graph=True, create_graph=False
        )[0]
        for xi, dpsi_dxi in zip(xis, (dpsi_dxs[..., i] for i in range(len(xis))))
    )

    Hpsi = (-0.5 * laplacian) + (E_pot.squeeze() * psi.squeeze()) # Laplacian is (1, N), while psi and E_pot are (N, 1, 1). 
                                                                               # squeeze() brings everything to (1, N)
    E_tot = torch.mean(Hpsi / psi.squeeze())
    return E_tot, psi, Hpsi
    