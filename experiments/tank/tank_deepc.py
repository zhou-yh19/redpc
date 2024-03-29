import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import sys
import os
import csv
import argparse
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../../"))
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from typing import List
from pydeepc import DeePC
from src.modules.pydeepc_l2_square import DeePC_QP
from src.modules.pydeepc_l12 import DeePC_l12
from experiments.utils import System, Data, split_data

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--horizon', type=int, default=40)
parser.add_argument('--lambda-g1', type=float, default=0)
parser.add_argument('--lambda-y1', type=float, default=0)
parser.add_argument('--lambda-g2', type=float, default=0)
parser.add_argument('--lambda-y2', type=float, default=0)
parser.add_argument('--q-weight', type=float, default=35)
parser.add_argument('--r-weight', type=float, default=1e-4)
parser.add_argument('--norm', type=int, default=12)
parser.add_argument('--process-std', type=float, default=0.01)
parser.add_argument('--measurement-std', type=float, default=0.1)
args = parser.parse_args()

# DeePC paramters
s = 1                       # How many steps before we solve again the DeePC problem
T_INI = 10                   # Size of the initial set of data
T_list = [i for i in range(200, 2001, 300)]             # Number of data points used to estimate the system
seed = args.seed
np.random.seed(seed)
Q_WEIGHT = args.q_weight
R_WEIGHT = args.r_weight
HORIZON = args.horizon                # Horizon length
LAMBDA_G1_REGULARIZER = args.lambda_g1
LAMBDA_Y1_REGULARIZER = args.lambda_y1
LAMBDA_G2_REGULARIZER = args.lambda_g2
LAMBDA_Y2_REGULARIZER = args.lambda_y2  
NORM = args.norm
EXPERIMENT_HORIZON = 100    # Total number of steps
EXP_NAME = f"process_std_{args.process_std}_measurement_std_{args.measurement_std}"
CASE_NAME = f"tank_deepc_{NORM}_{LAMBDA_G1_REGULARIZER}_{LAMBDA_Y1_REGULARIZER}_{LAMBDA_G2_REGULARIZER}_{LAMBDA_Y2_REGULARIZER}_{Q_WEIGHT}_{R_WEIGHT}_{HORIZON}_{seed}"
if not os.path.exists(os.path.join(file_path, f'deepc_exp/{EXP_NAME}')):
    os.makedirs(os.path.join(file_path, f'deepc_exp/{EXP_NAME}'))
if not os.path.exists(os.path.join(file_path, f'deepc_exp/{EXP_NAME}/{CASE_NAME}')):
    os.makedirs(os.path.join(file_path, f'deepc_exp/{EXP_NAME}/{CASE_NAME}'))
save_path = os.path.join(file_path, f'deepc_exp/{EXP_NAME}/{CASE_NAME}')

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

# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    y_ref = np.array([0.65, 0.77])
    Q = Q_WEIGHT * np.eye(P) # Weighting matrix for the output
    R = R_WEIGHT * np.eye(M) # Weighting matrix for the input
    # Note that u and y are matrices of size (horizon, M) and (horizon, P), respectively
    return cp.sum([cp.quad_form(y[i, :] - y_ref, Q) + cp.quad_form(u[i, :], R) for i in range(horizon)])

# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    constraints = [u >= -2, u <= 2, y >= -2, y <= 2]
    return constraints

fig, ax = plt.subplots(2,2, figsize=(15, 10))
plt.margins(x=0, y=0)

lq_cost_all = []
## used for LQ cost calculation
Q_ = np.kron(np.eye(EXPERIMENT_HORIZON), Q)
R_ = np.kron(np.eye(EXPERIMENT_HORIZON), R)
ref_ = np.tile(np.array([0.65, 0.77]), EXPERIMENT_HORIZON).flatten()

time_avg_list = []
time_all = []
violations_u = []
violations_y = []

# Simulate for different values of T
for T in T_list:
    print(f'Simulating with {T} initial samples...')
    sys.reset()
    # Load initial data and initialize DeePC
    data_u = np.load(os.path.join(file_path, f'data/{EXP_NAME}/data_traj_u.npy'))
    data_y = np.load(os.path.join(file_path, f'data/{EXP_NAME}/data_traj_y.npy'))
    data = Data(u = data_u[:T, :], y = data_y[:T, :])
    print(f'Loaded data with shape: {data.u.shape}, {data.y.shape}')
    Up, Uf, Yp, Yf = split_data(data, Tini = T_INI, horizon = HORIZON)
    if NORM == 2:
        deepc = DeePC_QP(data, Tini = T_INI, horizon = HORIZON)
        deepc.build_problem(
        Q = Q,
        R = R,
        ref = ref,
        u_upper = u_upper,
        u_lower = u_lower,
        y_upper = y_upper,
        y_lower = y_lower,
        lambda_g = LAMBDA_G2_REGULARIZER,
        lambda_y = LAMBDA_Y2_REGULARIZER,
    )
    elif NORM == 1:
        deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)
        deepc.build_problem(
        build_loss = loss_callback,
        build_constraints = constraints_callback,
        lambda_g = LAMBDA_G1_REGULARIZER,
        lambda_y = LAMBDA_Y1_REGULARIZER
    )
    elif NORM == 12:
        deepc = DeePC_l12(data, Tini = T_INI, horizon = HORIZON)
        deepc.build_problem(
        build_loss = loss_callback,
        build_constraints = constraints_callback,
        lambda_g1 = LAMBDA_G1_REGULARIZER,
        lambda_y1 = LAMBDA_Y1_REGULARIZER,
        lambda_g2 = LAMBDA_G2_REGULARIZER,
        lambda_y2 = LAMBDA_Y2_REGULARIZER
    )
    
    # Create initial data
    data_ini = Data(u = np.zeros((T_INI, 2)), y = np.zeros((T_INI, 2)))
    sys.reset(data_ini = data_ini)

    time_record = []

    for _ in range(EXPERIMENT_HORIZON // s):
        # Solve DeePC
        start = time.time()
        if NORM == 2:
            u_optimal, info = deepc.solve(data_ini = data_ini, warm_start=True, solver='OSQP', eps_abs=1e-12, eps_rel=1e-8, eps_prim_inf=1e-8, eps_dual_inf=1e-8)
        else:
            u_optimal, info = deepc.solve(data_ini = data_ini, warm_start=True, solver='MOSEK', verbose=False, mosek_params={'MSK_IPAR_NUM_THREADS': 1})
        end = time.time()
        time_record.append(end - start)

        # Apply optimal control input
        _ = sys.simulate(u = u_optimal[:s, :], process_std=args.process_std, measurement_std=args.measurement_std)

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    data = sys.get_all_samples()
    ax[0, 0].plot(data.y[T_INI:, 0])
    ax[0, 1].plot(data.y[T_INI:, 1], label=f'T={T}')
    ax[1, 0].plot(data.u[T_INI:, 0])
    ax[1, 1].plot(data.u[T_INI:, 1], label=f'T={T}')
    ax[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    time_avg_list.append(np.mean(time_record))
    time_all.append(time_record)

    ## calculate LQ cost
    data_y_ = data.y[T_INI:].flatten()
    data_u_ = data.u[T_INI:].flatten()
    lq_cost = (data_y_ - ref_).reshape(-1, 1).T @ Q_ @ (data_y_ - ref_).reshape(-1, 1) + data_u_.reshape(-1, 1).T @ R_ @ data_u_.reshape(-1, 1)
    lq_cost_all.append(lq_cost.item())
    violation_y = np.sum((data.y[T_INI:, 0] > 2) | (data.y[T_INI:, 0] < -2) | (data.y[T_INI:, 1] > 2) | (data.y[T_INI:, 1] < -2))
    violation_u = np.sum((data.u[T_INI:, 0] > 2) | (data.u[T_INI:, 0] < -2) | (data.u[T_INI:, 1] > 2) | (data.u[T_INI:, 1] < -2))
    violations_y.append(violation_y)
    violations_u.append(violation_u)


ax[0, 0].plot(np.tile(0.65, EXPERIMENT_HORIZON), label='ref_y1', c='k', linestyle='--')
ax[0, 1].plot(np.tile(0.77, EXPERIMENT_HORIZON), label='ref_y2', c='k', linestyle='--')

ax[0, 0].set_ylim(0., 0.85)
ax[0, 1].set_ylim(0., 0.95)


for i in range(2):
    ax[0, i].set_xlabel('t')
    ax[0, i].set_ylabel('y')
    ax[0, i].grid()
    ax[1, i].set_ylabel('u')
    ax[1, i].set_xlabel('t')
    ax[1, i].grid()
    ax[0, i].set_title(f'Closed loop - output signal $y_{i+1}$')
    ax[1, i].set_title(f'Closed loop - control signal $u_{i+1}$')

plt.savefig(os.path.join(save_path, 'tank_deepc.png'))

fig_time, ax_time = plt.subplots(1, 1)
ax_time.plot(T_list, time_avg_list, marker='o')
ax_time.set_xlabel('T')
ax_time.set_ylabel('Time (s)')
ax_time.set_title('Computation time')
plt.savefig(os.path.join(save_path, 'time_deepc.png'))

fig_cost, ax_cost = plt.subplots(1, 1)
ax_cost.plot(T_list, lq_cost_all, marker='o')
ax_cost.set_xlabel('T')
ax_cost.set_ylabel('LQ cost')
ax_cost.set_title('LQ cost')
plt.savefig(os.path.join(save_path, 'cost_deepc.png'))

fig_cost_time, ax_cost_time = plt.subplots(1, 1)
ax_cost_time.plot(time_avg_list, lq_cost_all, marker='o')
ax_cost_time.set_xlabel('Average Solving Time (s)')
ax_cost_time.set_ylabel('LQ cost')
ax_cost_time.set_title('LQ cost vs. Average Solving Time')
plt.savefig(os.path.join(save_path, 'cost_time_deepc.png'))


## Save the results to a CSV file
performance_csv_file_path = os.path.join(file_path, f'data/{EXP_NAME}/results_deepc_{NORM}.csv')

# Check if the CSV file exists to decide whether to write headers
write_headers = not os.path.isfile(performance_csv_file_path)

# Column headers for the CSV file
headers = ['Seed', 'Horizon', 'Lambda_G', 'Lambda_Y', 'Lambda_G_', 'Lambda_Y_', 'Q_Weight', 'R_Weight', 'Norm', 'T', 'Average Time (s)', 'LQ Cost', 'Violation_y', 'Violation_u']

# Data to be appended to the CSV file
data = [seed, HORIZON, LAMBDA_G1_REGULARIZER, LAMBDA_Y1_REGULARIZER, LAMBDA_G2_REGULARIZER, LAMBDA_Y2_REGULARIZER, Q_WEIGHT, R_WEIGHT, NORM]

# Append results to the CSV file
with open(performance_csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_headers:
        writer.writerow(headers)  # Write the headers if file doesn't exist
    for T, time_avg, cost, violation_y, violation_u in zip(T_list, time_avg_list, lq_cost_all, violations_y, violations_u):
        writer.writerow(data + [T, time_avg, cost, violation_y, violation_u])  # Append the new row with parameters and results

time_csv_file_path = os.path.join(file_path, f'data/{EXP_NAME}/time_deepc_{NORM}.csv')

# Check if the CSV file exists to decide whether to write headers
write_headers = not os.path.isfile(time_csv_file_path)

# Column headers for the CSV file
headers = [i for i in range(EXPERIMENT_HORIZON)]

# Data to be appended to the CSV file
data = time_all

# Append results to the CSV file
with open(time_csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_headers:
        writer.writerow(headers)  # Write the headers if file doesn't exist
    for time_record in time_all:
        writer.writerow(time_record)  # Append the new row with parameters and results