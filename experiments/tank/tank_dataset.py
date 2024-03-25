import numpy as np
import scipy.signal as scipysig
import os
import sys
import argparse
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../"))
from matplotlib import pyplot as plt
from utils import System, controllability_matrix, observability_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, default=500000, help="Number of samples")
parser.add_argument("--process-std", type=float, default=0., help="Standard deviation of process noise")
parser.add_argument("--measurement-std", type=float, default=0.1, help="Standard deviation of measurement noise")
parser.add_argument("--seed", type=int, default=2024, help="Random seed")

args = parser.parse_args()
N = args.num_samples
process_std = args.process_std
measurement_std = args.measurement_std
random_seed = args.seed

## model of tank system
A = np.array([
        [0.921, 0, 0.041, 0],
        [0, 0.918, 0, 0.033],
        [0, 0, 0.924, 0],
        [0, 0, 0, 0.937]])
B = np.array([[0.017, 0.001], [0.001, 0.023], [0, 0.061], [0.072, 0]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
D = np.zeros((C.shape[0], B.shape[1]))
n_in = 2
n_out = 2

## properties of the system
C_matrix, is_controllable = controllability_matrix(A, B)
O_matrix, is_observable = observability_matrix(A, C)
if is_controllable:
    print("The system is controllable.")
else:
    print("The system is not controllable.")
if is_observable:
    print("The system is observable.")
else:
    print("The system is not observable.")

## system
sys = System(scipysig.StateSpace(A, B, C, D, dt=1.))

## data generation
sys.reset()
np.random.seed(random_seed)
if process_std > 0:
    data = sys.simulate(u = np.random.uniform(-2, 2, size=(N, n_in)), process_std=process_std, measurement_std=measurement_std)

exp_name = f"process_std_{process_std}_measurement_std_{measurement_std}"
data_path = os.path.join(file_path, "data")
if not os.path.exists(data_path):
    os.makedirs(data_path)
exp_path = os.path.join(data_path, exp_name)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

## plot part of the data
sequence_to_plot = 500
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(data.u[:sequence_to_plot, 0])
ax[0, 0].set_xlabel("Time")
ax[0, 0].set_ylabel("u1")
ax[0, 1].plot(data.u[:sequence_to_plot, 1])
ax[0, 1].set_xlabel("Time")
ax[0, 1].set_ylabel("u2")
ax[1, 0].plot(data.y[:sequence_to_plot, 0])
ax[1, 0].set_xlabel("Time")
ax[1, 0].set_ylabel("y1")
ax[1, 1].plot(data.y[:sequence_to_plot, 1])
ax[1, 1].set_xlabel("Time")
ax[1, 1].set_ylabel("y2")

fig.suptitle(f"Tank system first {sequence_to_plot} samples")
plt.savefig(os.path.join(exp_path, "tank_system_original_data.png"))

print("initial data shape:")
print(f"u: {data.u.shape}")
print(f"y: {data.y.shape}")
np.save(os.path.join(exp_path, 'data_traj_u.npy'), data.u)
np.save(os.path.join(exp_path, 'data_traj_y.npy'), data.y)
