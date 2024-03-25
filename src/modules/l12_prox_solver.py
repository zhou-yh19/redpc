import torch
from torch import nn
from torch.nn import functional as F
from ..utils.torch_utils import bmv, bma, bsolve

class L12ProxSolver(nn.Module):
    """
    Solve the optimization problem with given constraints and iterative updates.
    """
    def __init__(self, device='cuda'):
        super(L12ProxSolver, self).__init__()
        self.device = device

    def soft_thresholding(self, x, alpha, D1, D2):
        """
        Apply the custom soft-thresholding operation based on D1 and D2.
        """
        return torch.where(x - alpha * torch.abs(D1) > 0,
                           (x - alpha * torch.abs(D1)) / (1 + 2 * alpha * (D2)**2),
                           torch.where(x + alpha * torch.abs(D1) < 0,
                                       (x + alpha * torch.abs(D1)) / (1 + 2 * alpha * (D2)**2),
                                       torch.zeros_like(x)))

    def forward(self, G, W, T, D1, D2, bs, alpha, max_iters):
        """
        Solve the optimization problem using iterative updates.

        Args:
        - G: torch.Tensor, shape (bs, m, n)
        - W: torch.Tensor, shape (bs, m, p)
        - T: torch.Tensor, shape (bs, p)
        - D1: torch.Tensor, shape (bs, n), diagonal elements for D1
        - D2: torch.Tensor, shape (bs, n), diagonal elements for D2
        - iters: int, number of iterations
        - alpha: float, step size
        - beta: float, step size
        - keep_iters: bool, if True, returns all iterates
        """
        m = G.shape[1]
        n = G.shape[2]
        p = W.shape[2]
        z = torch.zeros((bs, n), device=self.device)
        tau = torch.zeros((bs, p), device=self.device)
        xi = torch.zeros((bs, n), device=self.device)
        eta = torch.zeros((bs, p), device=self.device)
        G_ext = torch.cat((G, W), dim=2)

        I_GG = torch.eye(n + p, device=self.device).unsqueeze(0) - torch.bmm(torch.linalg.pinv(G_ext), G_ext)
        for _ in range(max_iters):
            sh = self.soft_thresholding(xi, alpha, D1, D2)

            temp = bmv(I_GG, torch.cat((2 * sh - xi, T), dim=1))
            z = temp[:, :n]
            tau = temp[:, n:]

            xi = xi + z - sh
            eta = eta + tau - (T + eta) / 2

        return z, tau

