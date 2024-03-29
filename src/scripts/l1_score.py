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
from src.modules.deepc_l1 import oracle_to_l1_batch, oracle_to_l1
from src.modules.l1_minimization_solver import L1MinimizationSolver
from multiprocessing import Pool

## parameters
parser = argparse.ArgumentParser()
parser.add_argument('--T-INI', type=int, default=10, help='initial time steps')
parser.add_argument('--HORIZON', type=int, default=40, help='prediction horizon')
parser.add_argument('--LENGTH', type=int, default=400, help='number of data points used to estimate the system')
parser.add_argument('--LAMBDA-G', type=float, default=1, help='regularization parameter for g')
parser.add_argument('--LAMBDA-Y', type=float, default=5e5, help='regularization parameter for sigma_y')
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

## compare the original formulation and our formulation to test formulation

if args.test:
    test_case = min(10, n_traj)
    g_oracle_original = np.zeros((test_case, n_g))
    g_oracle_ours = np.zeros((test_case, n_g))
    sigma_y_oracle_original = np.zeros((test_case, n_sigma_y))
    sigma_y_oracle_ours = np.zeros((test_case, n_sigma_y))
    result_oracle_original = np.zeros(test_case)
    result_oracle_ours = np.zeros(test_case)

    np.random.seed(random_seed)
    for i in range(test_case):
        # original formulation
        print('TEST SAMPLES {}/{}'.format(i + 1, test_case))
        test_index = np.random.choice(n_traj)
        g = cp.Variable(shape=(n_g), name='g')
        sigma_y = cp.Variable(shape=(n_sigma_y), name='sigma_y')
        constraints = [Up @ g == uini[test_index], Yp @ g == yini[test_index] + sigma_y, Uf @ g == u_pred[test_index], Yf @ g == y_pred[test_index]]
        objective = cp.Minimize(cp.norm(LAMBDA_G * g, p=1) + cp.norm(LAMBDA_Y * sigma_y, p=1))
        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver='ECOS')
        result_oracle_original[i] = result
        g_oracle_original[i] = g.value
        sigma_y_oracle_original[i] = sigma_y.value

        # our formulation
        x = cp.Variable(shape=(n_g + n_sigma_y), name='x')
        A, b = oracle_to_l1(lambda_g=LAMBDA_G, lambda_y=LAMBDA_Y, upred=u_pred[test_index], ypred=y_pred[test_index], Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, uini=uini[test_index], yini=yini[test_index])
        constraints = [A @ x == b]
        objective = cp.Minimize(cp.norm(x, p=1))
        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver='ECOS')
        result_oracle_ours[i] = result
        g_oracle_ours[i] = x.value[:n_g]
        sigma_y_oracle_ours[i] = x.value[n_g:]

    print(result_oracle_ours)
    print(result_oracle_original)
    print(f"max diff of result_oracle_original and result_oracle_ours: {np.max(np.abs(result_oracle_original - result_oracle_ours))}")
    print(f"max diff of g_oracle_original and g_oracle_ours: {np.max(np.abs(g_oracle_original - g_oracle_ours))}")
    print(f"max diff of sigma_y_oracle_original and sigma_y_oracle_ours: {np.max(np.abs(sigma_y_oracle_original - sigma_y_oracle_ours))}")

## compute the l1 score
def compute_trajectory_score(i):
    print(f'COMPUTE SAMPLES {i + 1}/{n_traj}')
    g = cp.Variable(shape=(n_g), name='g')
    sigma_y = cp.Variable(shape=(n_sigma_y), name='sigma_y')
    constraints = [Up @ g == uini[i], Yp @ g == yini[i] + sigma_y, Uf @ g == u_pred[i], Yf @ g == y_pred[i]]
    objective = cp.Minimize(cp.norm(LAMBDA_G * g, p=1) + cp.norm(LAMBDA_Y * sigma_y, p=1))
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver="ECOS")
    return result

def parallel_process(n_traj, num_processes=16):
    with Pool(processes=num_processes) as pool:
        scores = pool.map(compute_trajectory_score, range(n_traj))
    return np.array(scores)
dataset_score = parallel_process(n_traj)

## plotting
if TUNING:
    np.save(os.path.join(path, 'l1_score_tuning.npy'), dataset_score)
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
        np.save(os.path.join(path, 'l1_score_overlap.npy'), dataset_score)
        plt.savefig(os.path.join(path, 'l1_score_vs_std_overlap.png'))
    else:
        np.save(os.path.join(path, 'l1_score.npy'), dataset_score)
        plt.savefig(os.path.join(path, 'l1_score_vs_std.png'))