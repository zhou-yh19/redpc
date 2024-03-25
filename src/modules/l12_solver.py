import torch
from torch import nn
from torch.nn import functional as F
from ..utils.torch_utils import bmv, bma, bsolve

class L12Solver(nn.Module):
    """
    Solve the optimization problem with given constraints and iterative updates.
    """
    def __init__(self, device='cuda'):
        super(L12Solver, self).__init__()
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

    def projection(self, y, A, b):
        """
        Compute the projection pi(y) for batched inputs, efficiently handling
        the inversion and multiplication for batched matrices.
        
        Args:
        - y: Input tensor with shape (bs, n), where bs is batch size and n is the dimension.
        - A: Matrix A with shape (bs, m, n), representing the linear transformation.
        - b: Vector b with shape (bs, m), representing the constraint in Ax = b.

        Returns:
        - Tensor: The result of the projection operation on y.
        """
        # Compute A * A^T and its inverse for each batch
        AAT = torch.bmm(A, A.transpose(1, 2))  # Shape: (bs, m, m)
        ATA_inv = torch.linalg.inv(AAT)  # Shape: (bs, m, m)

        # Compute the correction term: A^T * (A A^T)^{-1} * (A y - b)
        # Note: Need to unsqueeze b to perform bmm correctly, then squeeze back
        Ay_minus_b = bmv(A, y) - b  # Shape: (bs, m)
        correction = bmv(A.transpose(1, 2), bmv(ATA_inv, Ay_minus_b))  # Shape: (bs, n)

        # Apply correction to y and return the result
        projected_y = y - correction
        return projected_y

    def forward(self, A, b, D1, D2, bs, iters=100, alpha=0.1, keep_iters=False):
        """
        Solve the optimization problem using iterative updates.

        Args:
        - A: torch.Tensor, shape (bs, m, n)
        - b: torch.Tensor, shape (bs, m)
        - D1: torch.Tensor, shape (bs, n), diagonal elements for D1
        - D2: torch.Tensor, shape (bs, n), diagonal elements for D2
        - iters: int, number of iterations
        - alpha: float, step size
        - beta: float, step size
        - keep_iters: bool, if True, returns all iterates
        """
        _, n = D1.shape
        _, m = b.shape
        x = torch.zeros((bs, n), dtype=torch.float32, device=self.device)
        z = torch.zeros((bs, n), dtype=torch.float32, device=self.device)
        iterates = torch.zeros(bs, iters, n + n, device=self.device) if keep_iters else None

        for k in range(iters):
            x_half = self.soft_thresholding(z, alpha, D1, D2)
            x = self.projection(2 * x_half - z, A, b)
            z = z + x - x_half

            if keep_iters:
                iterates[:, k, :n] = x.detach()
                iterates[:, k, n:] = z.detach()

        return iterates if keep_iters else x

