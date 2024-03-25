import torch
from torch import nn
from torch.nn import functional as F

from ..utils.torch_utils import bmv, bma, bsolve

class L1MinimizationSolver(nn.Module):
    """
    Solve L1 minimization problem:
    min ||x||_1 s.t. Ax = b
    where x in R^n, A in R^{m x n}, b in R^m
    """
    def __init__(self, device, n, m):
        super(L1MinimizationSolver, self).__init__()
        self.device = device
        self.n = n
        self.m = m

    @staticmethod
    def soft_thresholding(x, a):
        return F.relu(x - a) - F.relu(-x - a)

    def forward(self, A, b, bs, iters=1000, alpha=0.02, beta=0.02, keep_iters=False):
        """
        Solve L1 minimization problem using PDHG.

        Args:
        - A: torch.Tensor, shape (bs, m, n)
        - b: torch.Tensor, shape (bs, m)
        - iters: int, number of iterations
        - alpha: float, step size
        - beta: float, step size
        - keep_iters: bool, if True, return all iterates

        Returns:
        - x:
          If keep_iters is False, torch.Tensor, shape (bs, n), solution to the L1 minimization problem.
          If keep_iters is True, torch.Tensor, shape (bs, iters, n + m), all iterates of (primal, dual) variables.

        Iteration law is:
        x = sh(x - alpha * A' * u, alpha)
        u += beta * (A * (2 * x - x_old) - b)
        where x, u are primal and dual variables, respectively;
        sh is the soft thresholding operator:
        sh(x, a) = max(0, |x| - a) * sign(x)

        Caveat: convergence rate not optimized; may need a lot of iterations to get accurate solution.
        """
        iterates = torch.zeros(bs, iters, self.n + self.m, device=self.device) if keep_iters else None
        x = torch.zeros((bs, self.n), dtype=torch.float32, device=self.device)
        u = torch.zeros((bs, self.m), dtype=torch.float32, device=self.device)
        for k in range(iters):
            if keep_iters:
                iterates[:, k, :self.n] = x.detach()
                iterates[:, k, self.n:] = u.detach()
            x_old = x.clone()
            x = self.soft_thresholding(x_old - alpha * bmv(A.transpose(1, 2), u), alpha)
            u = u + beta * (bmv(A, 2 * x - x_old) - b)

        if keep_iters:
            return iterates
        else:
            return x

