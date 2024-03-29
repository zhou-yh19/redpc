import torch
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../../"))
from pydeepc.utils import Data, split_data
from multiprocessing import Pool

## parameters
parser = argparse.ArgumentParser()
parser.add_argument('--T-INI', type=int, default=10, help='initial time steps')
parser.add_argument('--HORIZON', type=int, default=40, help='prediction horizon')
parser.add_argument('--LENGTH', type=int, default=400, help='number of data points used to estimate the system')
parser.add_argument('--LAMBDA-G', type=float, default=1, help='regularization parameter for g')
parser.add_argument('--LAMBDA-Y', type=float, default=5e5, help='regularization parameter for sigma_y')
parser.add_argument('--LAMBDA-G-', type=float, default=0, help='regularization parameter for g')
parser.add_argument('--LAMBDA-Y-', type=float, default=0, help='regularization parameter for sigma_y')
parser.add_argument('--n-in', type=int, default=2, help='number of input channels')
parser.add_argument('--n-out', type=int, default=2, help='number of output channels')
parser.add_argument('--overlap', action="store_true", help="Enable overlap feature")
parser.add_argument('--system', type=str, default='tank', help='system name')
parser.add_argument("--process-std", type=float, default=0.01)
parser.add_argument("--measurement-std", type=float, default=0.1)
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--test', action="store_true", help="Enable test mode")
parser.add_argument('--tuning', action="store_true", help="Enable tuning mode")

args = parser.parse_args()
T_INI = args.T_INI
HORIZON = args.HORIZON
LENGTH = args.LENGTH
LAMBDA_G = args.LAMBDA_G
LAMBDA_Y = args.LAMBDA_Y
LAMBDA_G_ = args.LAMBDA_G_
LAMBDA_Y_ = args.LAMBDA_Y_
n_in = args.n_in
n_out = args.n_out
OVERLAP = args.overlap
TUNING = args.tuning
exp_name = f"process_std_{args.process_std}_measurement_std_{args.measurement_std}"
path = os.path.join(file_path, f"../../experiments/{args.system}/data/{exp_name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_g = LENGTH - (T_INI + HORIZON) + 1
n_sigma_y = T_INI * n_out
random_seed = 2024

## data load
data_u = np.load(os.path.join(path, 'data_traj_u.npy'))[:LENGTH]
data_y = np.load(os.path.join(path, 'data_traj_y.npy'))[:LENGTH]
Up, Uf, Yp, Yf = split_data(Data(data_u, data_y), T_INI, HORIZON)
print(f"Up: {Up.shape}, Uf: {Uf.shape}, Yp: {Yp.shape}, Yf: {Yf.shape}")
if TUNING:
    dataset_traj = np.load(os.path.join(path, 'traj4learning_tuning.npy'))
    print(f"dataset_traj_tuning: {dataset_traj.shape}")
elif OVERLAP:
    dataset_traj = np.load(os.path.join(path, 'traj4learning_overlap.npy'))
    print(f"dataset_traj_overlap: {dataset_traj.shape}")
else:
    dataset_traj = np.load(os.path.join(path, 'traj4learning.npy'))
    print(f"dataset_traj: {dataset_traj.shape}")

uini = dataset_traj[:, :T_INI * n_in]
u_pred = dataset_traj[:, T_INI * n_in: (T_INI + HORIZON) * n_in]
yini = dataset_traj[:, (T_INI + HORIZON) * n_in: (T_INI + HORIZON) * n_in + T_INI * n_out]
y_pred = dataset_traj[:, (T_INI + HORIZON) * n_in + T_INI * n_out:]
n_traj = dataset_traj.shape[0]

## compute the l12 score
def compute_trajectory_score(i):
    print(f'COMPUTE SAMPLES {i + 1}/{n_traj}')
    g = cp.Variable(shape=(n_g), name='g')
    sigma_y = cp.Variable(shape=(n_sigma_y), name='sigma_y')
    constraints = [Up @ g == uini[i], Yp @ g == yini[i] + sigma_y, Uf @ g == u_pred[i], Yf @ g == y_pred[i]]
    objective = cp.Minimize(LAMBDA_G * cp.norm(g, p=1) + LAMBDA_Y * cp.norm(sigma_y, p=1) + LAMBDA_G_ * cp.norm(g, p=2) ** 2 + LAMBDA_Y_ * cp.norm(sigma_y, p=2) ** 2)
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver="MOSEK", mosek_params={'MSK_IPAR_NUM_THREADS': 1})
    return result

def parallel_process(n_traj, num_processes=100):
    with Pool(processes=num_processes) as pool:
        scores = pool.map(compute_trajectory_score, range(n_traj))
    return np.array(scores)
dataset_score = parallel_process(n_traj)

## plotting
if TUNING:
    np.save(os.path.join(path, 'l12_score_tuning.npy'), dataset_score)
else:
    if OVERLAP:
        dataset_std = np.load(os.path.join(path, 'traj4learning_std_overlap.npy'))
    else:
        dataset_std = np.load(os.path.join(path, 'traj4learning_std.npy'))
    fig = plt.figure()
    plt.scatter(dataset_std, dataset_score, label="original data", s=0.1)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Standard deviation')
    plt.ylabel('Score')
    plt.title('Score vs. Standard deviation')
    plt.grid()

    if OVERLAP:
        np.save(os.path.join(path, 'l12_score_overlap.npy'), dataset_score)
        plt.savefig(os.path.join(path, 'l12_score_vs_std_overlap.png'))
    else:
        np.save(os.path.join(path, 'l12_score.npy'), dataset_score)
        plt.savefig(os.path.join(path, 'l12_score_vs_std.png'))

    plt.xscale('log')
    plt.yscale('log')
    if OVERLAP:
        plt.savefig(os.path.join(path, 'l12_score_vs_std_log_overlap.png'))
    else:
        plt.savefig(os.path.join(path, 'l12_score_vs_std_log.png'))