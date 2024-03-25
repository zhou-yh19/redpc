import os
import sys
import argparse
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../"))
from utils import create_hankel_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--process-std", type=float, default=0., help="Standard deviation of the process noise")
parser.add_argument("--measurement-std", type=float, default=0.1, help="Standard deviation of the measurement noise")
parser.add_argument("--trajectory-length", type=int, default=50, help="Length of the trajectory")

args = parser.parse_args()
process_std = args.process_std
measurement_std = args.measurement_std
T = args.trajectory_length

exp_name = f"process_std_{process_std}_measurement_std_{measurement_std}"
data_path = os.path.join(file_path, "data")
exp_path = os.path.join(data_path, exp_name)

original_u = np.load(os.path.join(exp_path, 'data_traj_u.npy'))
original_y = np.load(os.path.join(exp_path, 'data_traj_y.npy'))
assert original_u.shape[0] == original_y.shape[0]
N = original_u.shape[0]
n_in = original_u.shape[1]
n_out = original_y.shape[1]

Hu = create_hankel_matrix(original_u, T)
Hy = create_hankel_matrix(original_y, T)
trajectory_overlap = np.concatenate((Hu, Hy)).transpose()
print(f"trajectory overlap shape: {trajectory_overlap.shape}")
np.save(os.path.join(exp_path, 'trajectory_overlap.npy'), trajectory_overlap)

trajectory = np.zeros((N // T, T * (n_in + n_out)))
for idx in range(N // T):
    trajectory[idx, :T * n_in] = original_u[idx * T:(idx + 1) * T, :].flatten()
    trajectory[idx, T * n_in:] = original_y[idx * T:(idx + 1) * T, :].flatten()
print(f"trajectory non-overlap shape: {trajectory.shape}")
np.save(os.path.join(exp_path, 'trajectory.npy'), trajectory)