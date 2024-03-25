import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import sys
import os
import csv
import torch
import argparse
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../../"))
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from typing import List
from src.modules.pyredpc_l2_square import RedPC
from experiments.utils import System, Data, split_data
from src.utils.torch_utils import make_psd

parser = argparse.ArgumentParser()
parser.add_argument('--training-seed', type=int, default=2024)
parser.add_argument('--redpc-seed', type=int, default=2024)
parser.add_argument('--horizon', type=int, default=40)
parser.add_argument('--q-weight', type=float, default=35)
parser.add_argument('--r-weight', type=float, default=1e-4)
parser.add_argument('--norm', type=int, default=2)
parser.add_argument('--approximator', type=str, default='QP', help='approximator name')
parser.add_argument('--trajectory-size', type=int, default=400)
parser.add_argument('--m', type=int, default=48)
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--iters', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument("--overlap", action="store_true", help="Enable overlap feature (default: %(default)s)")
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--system', type=str, default='tank', help="system name")
parser.add_argument('--process-std', type=float, default=0.01)
parser.add_argument('--measurement-std', type=float, default=0.1)
args = parser.parse_args()

# RedPC paramters
s = 1                       # How many steps before we solve again the RedPC problem
T_INI = 10                   # Size of the initial set of data
T = args.trajectory_size              # Number of trajectories to estimate the system
Q_WEIGHT = args.q_weight
R_WEIGHT = args.r_weight
HORIZON = args.horizon                # Horizon length
norm = args.norm
m = args.m
n = args.n
iters = args.iters
approximator_name = args.approximator
training_seed = args.training_seed
redpc_seed = args.redpc_seed
np.random.seed(redpc_seed)
overlap = args.overlap
epochs = args.epochs
LR = args.lr
process_std = args.process_std
measurement_std = args.measurement_std
exp_name = f"process_std_{process_std}_measurement_std_{measurement_std}"

EXPERIMENT_HORIZON = 100    # Total number of steps

n_in = 2
n_out = 2

case_name = f"runs/{exp_name}/{approximator_name}_iters_{iters}_nqp_{n}_mqp_{m}_lr_{LR}_seed_{training_seed}_overlap_{overlap}_norm_{norm}"
save_path = os.path.join(file_path, case_name)
model_path = os.path.join(save_path, f'checkpoint_{epochs - 1}.pth')
model = torch.load(model_path)
state_dict = model['state_dict']
print(state_dict.keys())
G = state_dict['G'].cpu().numpy().reshape(m, n)
H = state_dict['H'].cpu().numpy().reshape(m, (T_INI + HORIZON) * (n_in + n_out))
redpc_params = {
    'n_u': n_in,
    'n_y': n_out,
    'n': n,
    'G': G,
    'H': H
}
if approximator_name == 'QP':
    P_param = state_dict['P_param']
    Pinv = make_psd(P_param.unsqueeze(0)).cpu().numpy().reshape(n, n)
    redpc_params['Pinv'] = Pinv
elif approximator_name == 'L12' or approximator_name == 'L12Prox':
    D1 = state_dict['D1'].cpu().numpy().reshape(n)
    D2 = state_dict['D2'].cpu().numpy().reshape(n)
    redpc_params['D1'] = D1
    redpc_params['D2'] = D2

# model of example
A = np.array([
        [0.921, 0, 0.041, 0],
        [0, 0.918, 0, 0.033],
        [0, 0, 0.924, 0],
        [0, 0, 0, 0.937]])
B = np.array([[0.017, 0.001], [0.001, 0.023], [0, 0.061], [0.072, 0]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
D = np.zeros((C.shape[0], B.shape[1]))

sys = System(scipysig.StateSpace(A, B, C, D, dt=1.))

## LQ cost and box constraints
Q = Q_WEIGHT * np.eye(2)
R = R_WEIGHT * np.eye(2)
ref = np.tile(np.array([0.65, 0.77]), (HORIZON)).flatten()
u_upper = 2 * np.ones(2)
u_lower = -2 * np.ones(2)
y_upper = 2 * np.ones(2)
y_lower = -2 * np.ones(2)

# Define the loss function for RedPC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    y_ref = np.array([0.65, 0.77])
    Q = Q_WEIGHT * np.eye(P) # Weighting matrix for the output
    R = R_WEIGHT * np.eye(M) # Weighting matrix for the input
    # Note that u and y are matrices of size (horizon, M) and (horizon, P), respectively
    return cp.sum([cp.quad_form(y[i, :] - y_ref, Q) + cp.quad_form(u[i, :], R) for i in range(horizon)])

# Define the constraints for RedPC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    constraints = [u >= -2, u <= 2, y >= -2, y <= 2]
    return constraints

fig, ax = plt.subplots(2,2, figsize=(15, 10))
plt.margins(x=0, y=0)

## used for LQ cost calculation
Q_ = np.kron(np.eye(EXPERIMENT_HORIZON), Q)
R_ = np.kron(np.eye(EXPERIMENT_HORIZON), R)
ref_ = np.tile(np.array([0.65, 0.77]), EXPERIMENT_HORIZON).flatten()

# Simulate for different values of T
print(f'Simulating with {T} initial samples...')
sys.reset()

# Initialize RedPC
redpc = RedPC(approximator_name=approximator_name, param=redpc_params, Tini = T_INI, horizon = HORIZON)
redpc.build_problem(
    build_loss = loss_callback,
    build_constraints = constraints_callback
    )

# Create initial data
data_ini = Data(u = np.zeros((T_INI, 2)), y = np.zeros((T_INI, 2)))
sys.reset(data_ini = data_ini)

time_record = []

for _ in range(EXPERIMENT_HORIZON // s):
    # Solve RedPC
    start = time.time()
    if approximator_name == 'QP':
        u_optimal, info = redpc.solve(data_ini = data_ini, warm_start=True, solver='OSQP', eps_abs=1e-12, eps_rel=1e-8, eps_prim_inf=1e-8, eps_dual_inf=1e-8, numberThreads = 1)
    else:
        u_optimal, info = redpc.solve(data_ini = data_ini, warm_start=True, solver='MOSEK', verbose=False, mosek_params={'MSK_IPAR_NUM_THREADS': 1})
    end = time.time()

    time_record.append(end - start)

    # Apply optimal control input
    _ = sys.simulate(u = u_optimal[:s, :], process_std=args.process_std, measurement_std=args.measurement_std)

    # Fetch last T_INI samples
    data_ini = sys.get_last_n_samples(T_INI)

data = sys.get_all_samples()
ax[0, 0].plot(data.y[T_INI:, 0])
ax[0, 1].plot(data.y[T_INI:, 1])
ax[1, 0].plot(data.u[T_INI:, 0])
ax[1, 1].plot(data.u[T_INI:, 1])

## calculate LQ cost
data_y_ = data.y[T_INI:].flatten()
data_u_ = data.u[T_INI:].flatten()
lq_cost = (data_y_ - ref_).reshape(-1, 1).T @ Q_ @ (data_y_ - ref_).reshape(-1, 1) + data_u_.reshape(-1, 1).T @ R_ @ data_u_.reshape(-1, 1)
violation_y = np.sum((data.y[:, 0] < -2) | (data.y[:, 0] > 2) | (data.y[:, 1] < -2) | (data.y[:, 1] > 2))
violation_u = np.sum((data.u[:, 0] < -2) | (data.u[:, 0] > 2) | (data.u[:, 1] < -2) | (data.u[:, 1] > 2))

ax[0, 0].plot(np.tile(0.65, EXPERIMENT_HORIZON), label='ref_y1', c='k', linestyle='--')
ax[0, 1].plot(np.tile(0.77, EXPERIMENT_HORIZON), label='ref_y2', c='k', linestyle='--')

for i in range(2):
    ax[0, i].set_xlabel('t')
    ax[0, i].set_ylabel('y')
    ax[0, i].grid()
    ax[1, i].set_ylabel('u')
    ax[1, i].set_xlabel('t')
    ax[1, i].grid()
    ax[0, i].set_title(f'Closed loop - output signal $y_{i+1}$')
    ax[1, i].set_title(f'Closed loop - control signal $u_{i+1}$')
## mark computation time, lq cost and trajectory length on the plot
ax[0, 0].text(0, 0.75, f'Average Time: {np.mean(time_record):.4f} s', fontsize=12, color='black')
ax[0, 0].text(50, 0.75, f'LQ Cost: {lq_cost[0, 0]:.4f}', fontsize=12, color='black')
ax[0, 0].text(100, 0.75, f'Trajectory Length: {T}', fontsize=12, color='black')
plt.savefig(os.path.join(save_path, 'tank_redpc.png'))

## Save the results to a CSV file
performance_csv_file_path = os.path.join(file_path,  f'data/{exp_name}/results_redpc_{norm}.csv')
time_csv_file_path = os.path.join(file_path,  f'data/{exp_name}/time_redpc_{norm}.csv')

# Check if the CSV file exists to decide whether to write headers
write_headers = not os.path.isfile(performance_csv_file_path)

# Column headers for the CSV file
headers = ['approximator', 'iters', 'n_qp', 'm_qp', 'lr', 'seed', 'redpc_seed', 'overlap', 'T', 'time_avg', 'cost', 'violation_y', 'violation_u']

# Data to be appended to the CSV file
data = [approximator_name, iters, n, m, LR, training_seed, redpc_seed, overlap]

# Append results to the CSV file
with open(performance_csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_headers:
        writer.writerow(headers)  # Write the headers if file doesn't exist
    writer.writerow(data + [T, np.mean(time_record), lq_cost[0, 0], violation_y, violation_u])

# Check if the CSV file exists to decide whether to write headers
write_headers = not os.path.isfile(time_csv_file_path)

# Column headers for the CSV file
headers = [i for i in range(EXPERIMENT_HORIZON)]

# Data to be appended to the CSV file
data = time_record

# Append results to the CSV file
with open(time_csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_headers:
        writer.writerow(headers)  # Write the headers if file doesn't exist
    writer.writerow(data)  # Write the data