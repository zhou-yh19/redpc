import torch
import torch.nn as nn

from .qp_solver import QPSolver
from .l1_minimization_solver import L1MinimizationSolver
from .l12_solver import L12Solver
from .l12_prox_solver import L12ProxSolver
from ..utils.torch_utils import make_psd, bmv, bsolve, bvv

class DeepCScoreApproximator(nn.Module):
    """Predict the score function of DeepC using a function approximator.

    Input: (bs, N * (n_in + n_out)), a batch of trajectories of length N; each trajectory is arranged in the form of (u_1, ..., u_N, y_1, ..., y_N).
    Output: (bs,), a batch of scalar scores.

    Function approximator can be one of the following:
    - MLP
    - QP
    - L1 minimization
    - L1/L2 minimization
    - L1/L2 minimization with proximal operator
    """

    def __init__(self, n_in, n_out, N, approximator_name, approximator_size, device, iters=10):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.N = N
        self.trajectory_size = N * (n_in + n_out)
        self.approximator_name = approximator_name
        self.approximator_size = approximator_size
        self.device = device

        if approximator_name == 'MLP':
            hidden_sizes = approximator_size
            layers = [nn.Linear(self.trajectory_size, hidden_sizes[0]), nn.ReLU()]
            for i in range(1, len(hidden_sizes)):
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[-1], 1))
            self.approximator = nn.Sequential(*layers).to(device)
        elif approximator_name == 'QP':
            m_qp, n_qp = approximator_size
            self.qp_iters = iters
            num_P_param = n_qp * (n_qp + 1) // 2
            self.P_param = nn.Parameter(torch.randn((num_P_param,), device=device))
            self.G = nn.Parameter(torch.randn((m_qp, n_qp), device=device))
            self.H = nn.Parameter(torch.randn((m_qp, self.trajectory_size), device=device))
            # self.Wq = nn.Parameter(torch.randn((n_qp, self.trajectory_size), device=device))
            self.m_qp, self.n_qp = m_qp, n_qp
            self.qp_solver = QPSolver(device, n_qp, m_qp, buffered=True)
            def qp_approximator(trajectory):
                bs = trajectory.shape[0]
                Pinv = make_psd(self.P_param.unsqueeze(0))
                # q = bmv(self.Wq.unsqueeze(0), trajectory)
                q = torch.zeros((bs, n_qp), device=device)
                H = self.G.unsqueeze(0)
                b = bmv(self.H.unsqueeze(0), trajectory)
                Xs, primal_sols, residuals = self.qp_solver(q, b, Pinv=Pinv, H=H, iters=self.qp_iters, return_residuals=True)
                solution = primal_sols[:, -1, :]
                # return optimal value
                return 0.5 * (solution * bsolve(Pinv, solution)).sum(dim=1)
                # return 0.5 * (solution * bsolve(Pinv, solution)).sum(dim=1) + bvv(q, solution).squeeze(1)
            self.approximator = qp_approximator
        elif approximator_name == 'L1':
            m_l1, n_l1 = approximator_size
            self.l1_iters = iters
            self.G = nn.Parameter(torch.randn((m_l1, n_l1), device=device))
            self.H = nn.Parameter(torch.randn((m_l1, self.trajectory_size), device=device))
            self.l1_solver = L1MinimizationSolver(device, n_l1, m_l1)
            def l1_approximator(trajectory):
                bs = trajectory.shape[0]
                A = self.G.unsqueeze(0)
                b = bmv(self.H.unsqueeze(0), trajectory)
                x = self.l1_solver(A, b, bs, iters=self.l1_iters)
                return x.abs().sum(dim=1)
            self.approximator = l1_approximator
        elif approximator_name == 'L12':
            m_l12, n_l12 = approximator_size
            self.l12_iters = iters
            self.G = nn.Parameter(torch.randn((m_l12, n_l12), device=device))
            self.H = nn.Parameter(torch.randn((m_l12, self.trajectory_size), device=device))
            self.D1 = nn.Parameter(torch.randn((n_l12,), device=device))
            self.D2 = nn.Parameter(torch.randn((n_l12,), device=device))
            self.l12_solver = L12Solver(device)
            def l12_approximator(trajectory):
                bs = trajectory.shape[0]
                A = self.G.unsqueeze(0)
                b = bmv(self.H.unsqueeze(0), trajectory)
                D1 = self.D1.unsqueeze(0)
                D2 = self.D2.unsqueeze(0)
                x = self.l12_solver(A, b, D1, D2, bs, iters=self.l12_iters, alpha=0.1)
                return torch.abs(D1 * x).sum(dim=1) + torch.square(D2 * x).sum(dim=1)
            self.approximator = l12_approximator
        elif approximator_name == 'L12Prox':
            m_l12, n_l12 = approximator_size
            self.l12_iters = iters
            self.G = nn.Parameter(torch.randn((m_l12, n_l12), device=device))
            self.H = nn.Parameter(torch.randn((m_l12, self.trajectory_size), device=device))
            self.D1 = nn.Parameter(torch.randn((n_l12,), device=device))
            self.D2 = nn.Parameter(torch.randn((n_l12,), device=device))
            self.l12_solver = L12ProxSolver(device)
            def l12_prox_approximator(trajectory):
                bs = trajectory.shape[0]
                G = self.G.unsqueeze(0)
                H = self.H.unsqueeze(0)
                D1 = self.D1.unsqueeze(0)
                D2 = self.D2.unsqueeze(0)
                z, tau = self.l12_solver(G, H, trajectory, D1, D2, bs, alpha=1., max_iters=self.l12_iters)
                return tau
            self.approximator = l12_prox_approximator
        else :
            raise ValueError(f"Unknown approximator name: {approximator_name}") 

    def forward(self, trajectory):
        return self.approximator(trajectory)
