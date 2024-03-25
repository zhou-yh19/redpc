import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt
import sys
import csv
import os
import argparse
import time
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../../"))
from experiments.utils import System

argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=2024)
argparser.add_argument("--process-std", type=float, default=0.01)
argparser.add_argument("--measurement-std", type=float, default=0.1)
args = argparser.parse_args()
seed = args.seed
process_std = args.process_std
measurement_std = args.measurement_std

EXP_NAME = f"process_std_{process_std}_measurement_std_{measurement_std}"
CASE_NAME = f"tank_mpc_{seed}"
if not os.path.exists(os.path.join(file_path, f'mpc')):
    os.makedirs(os.path.join(file_path, f'mpc'))
if not os.path.exists(os.path.join(file_path, f'mpc/{EXP_NAME}')):
    os.makedirs(os.path.join(file_path, f'mpc/{EXP_NAME}'))
mpc_path = os.path.join(file_path, f'mpc/{EXP_NAME}')
save_path = os.path.join(file_path, f'data/{EXP_NAME}')

# System matrices
A = np.array([
    [0.921, 0, 0.041, 0],
    [0, 0.918, 0, 0.033],
    [0, 0, 0.924, 0],
    [0, 0, 0, 0.937]])
B = np.array([[0.017, 0.001], [0.001, 0.023], [0, 0.061], [0.072, 0]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
D = np.zeros((C.shape[0], B.shape[1]))

sys = System(scipysig.StateSpace(A, B, C, D, dt=1.))

# Prediction horizon and bounds
N = 20
u_lower_bound = -2
u_upper_bound = 2
y_lower_bound = -2
y_upper_bound = 2

# Weights
Q_cost = 35 * np.eye(C.shape[0])  # Output deviation weight
R_cost = 1e-4 * np.eye(B.shape[1])  # Control input weight

# Initial state
x0 = cp.Parameter(A.shape[0])

# Desired output trajectory over the horizon
y_ref = np.array([0.65, 0.77])
y_ref_matrix = np.tile(y_ref, (N, 1)).T

# Decision variables
X = cp.Variable((A.shape[0], N+1))
U = cp.Variable((B.shape[1], N))
Y = cp.Variable((C.shape[0], N))

# experiments horizon
time_record = []
EXPERIMENT_HORIZON = 100

for i in range(EXPERIMENT_HORIZON):
    # Objective and constraints
    objective = 0
    x0 = sys.x0
    constraints = [X[:,0] == x0]
    for t in range(N):
        # Calculate output based on current state and input
        current_Y = C @ X[:,t] + D @ U[:,t]
        
        # System dynamics
        constraints += [
            X[:,t+1] == A @ X[:,t] + B @ U[:,t],
            U[:,t] >= u_lower_bound,
            U[:,t] <= u_upper_bound,
            current_Y >= y_lower_bound,
            current_Y <= y_upper_bound,
        ]
        
        # Update the objective function
        objective += cp.quad_form(current_Y - y_ref_matrix[:,t], Q_cost) + cp.quad_form(U[:,t], R_cost)

    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    start_time = time.time()
    problem.solve(warm_start=True, solver='MOSEK', verbose=False, mosek_params={'MSK_IPAR_NUM_THREADS': 1})
    end_time = time.time()

    u = U.value[:,0].reshape(1, -1)

    _ = sys.simulate(u = u, process_std=process_std, measurement_std=measurement_std)

    time_record.append(end_time - start_time)

fig, ax = plt.subplots(2,2, figsize=(15, 10))
plt.margins(x=0, y=0)
data = sys.get_all_samples()
ax[0, 0].plot(data.y[:, 0])
ax[0, 1].plot(data.y[:, 1])
ax[1, 0].plot(data.u[:, 0])
ax[1, 1].plot(data.u[:, 1])

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

Q_ = np.kron(np.eye(EXPERIMENT_HORIZON), Q_cost)
R_ = np.kron(np.eye(EXPERIMENT_HORIZON), R_cost)
ref_ = np.tile(np.array([0.65, 0.77]), EXPERIMENT_HORIZON).flatten()

## calculate LQ cost, time and violation
data_y_ = data.y[:].flatten()
data_u_ = data.u[:].flatten()
lq_cost = (data_y_ - ref_).reshape(-1, 1).T @ Q_ @ (data_y_ - ref_).reshape(-1, 1) + data_u_.reshape(-1, 1).T @ R_ @ data_u_.reshape(-1, 1)
time = np.mean(time_record)
violation_y = np.sum((data.y[:, 0] < y_lower_bound) | (data.y[:, 0] > y_upper_bound) | (data.y[:, 1] < y_lower_bound) | (data.y[:, 1] > y_upper_bound))
violation_u = np.sum((data.u[:, 0] < u_lower_bound) | (data.u[:, 0] > u_upper_bound) | (data.u[:, 1] < u_lower_bound) | (data.u[:, 1] > u_upper_bound))

# save fig
plt.savefig(os.path.join(mpc_path, f"{CASE_NAME}.png"))

## Save the results to a CSV file
performance_csv_file_path = os.path.join(save_path, f'results_mpc.csv')
time_csv_file_path = os.path.join(save_path, f'time_mpc.csv')

# Check if the CSV file exists to decide whether to write headers
write_headers = not os.path.isfile(performance_csv_file_path)

# Column headers for the CSV file
headers = ['Seed', 'Process Std', 'Measurement Std', 'Time', 'LQ Cost', 'Violation_y, Violation_u']

# Data to be appended to the CSV file
data = [seed, process_std, measurement_std, time, lq_cost.item(), violation_y, violation_u]

# Append results to the CSV file
with open(performance_csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_headers:
        writer.writerow(headers)  # Write the headers if file doesn't exist
    writer.writerow(data)  # Write the data

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
