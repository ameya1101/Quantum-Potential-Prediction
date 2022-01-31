import torch

class Hermite(torch.autograd.Function):
    """
    Implements 1D Hermite polynomials of the second kind. Used
    to implement the wavefunction for the harmonic oscillator.

    Forward pass uses the recurrence relation:
        H_{n+1}(x) = 2 * (x * H_{n}(x) - n * H_{n-1}(x))
    Backward pass uses the recurrence relation:
        H'_{n}(x) = 2 * n * H_{n-1}(x)
    """

    @staticmethod
    def forward(ctx, x, degree):
        initial = [lambda x: torch.ones_like(x), lambda x: 2*x]
        polys = [initial[0](x), initial[1](x)]
        recurrence = lambda p1, p2, n, x: (torch.multiply(x, p1) - n * p2) * 2

        if degree == 0 or degree == 1:
            ctx.poly, ctx.degree = polys[-2], degree
            return polys[degree]
        for i in range(1, degree):
            polys.append(recurrence(polys[-1], polys[-2], i, x))
        ctx.poly, ctx.degree = polys[-2], degree

        return polys[-1]
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.poly * ctx.degree * 2
        return grad_input, None
    