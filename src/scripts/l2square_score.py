import torch
from qpth.qp import QPFunction
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../../"))
from pydeepc.utils import Data, split_data
from src.modules.deepc_qp import oracle_to_qp, oracle_to_qp_batch

## parameters
parser = argparse.ArgumentParser()
parser.add_argument('--T-INI', type=int, default=10, help='initial time steps')
parser.add_argument('--HORIZON', type=int, default=40, help='prediction horizon')
parser.add_argument('--LENGTH', type=int, default=400, help='number of data points used to estimate the system')
parser.add_argument('--LAMBDA-G', type=float, default=30, help='regularization parameter for the control input')
parser.add_argument('--LAMBDA-Y', type=float, default=1e6, help='regularization parameter for the output')
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

if args.test:
    ## compare osqp and qpth to check the correctness of the solver
    test_case = min(10, n_traj)
    np.random.seed(random_seed)
    test_index = np.random.choice(n_traj, test_case, replace=False)
    ## test_index = np.arange(test_case)
    g_oracle_osqp = np.zeros((test_case, n_g))
    g_oracle_qpth = np.zeros((test_case, n_g))
    result_oracle_osqp = np.zeros(test_case)
    result_oracle_qpth = np.zeros(test_case)

    ## osqp part
    for i in range(test_case):
        print('TEST SAMPLES {}/{}'.format(i + 1, test_case))
        qp_Q, qp_p, qp_A, qp_b, constant = oracle_to_qp(lambda_g=LAMBDA_G, lambda_y=LAMBDA_Y, upred=u_pred[test_index[i]], ypred=y_pred[test_index[i]], Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, uini=uini[test_index[i]], yini=yini[test_index[i]])
        g = cp.Variable(shape=(LENGTH - (T_INI + HORIZON) + 1), name='g')
        constraints = [qp_A @ g == qp_b]
        objective = cp.Minimize(1 / 2 * cp.quad_form(g, qp_Q) + qp_p @ g)
        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver='OSQP', eps=1e-8)
        result_oracle_osqp[i] = result + constant
        g_oracle_osqp[i] = g.value

    ## qpth part
    LAMBDA_G = torch.tensor(LAMBDA_G, dtype=torch.float64).to(device)
    LAMBDA_Y = torch.tensor(LAMBDA_Y, dtype=torch.float64).to(device)
    Up = torch.tensor(Up, dtype=torch.float64).to(device)
    Uf = torch.tensor(Uf, dtype=torch.float64).to(device)
    Yp = torch.tensor(Yp, dtype=torch.float64).to(device)
    Yf = torch.tensor(Yf, dtype=torch.float64).to(device)
    uini = torch.tensor(uini, dtype=torch.float64).to(device)
    yini = torch.tensor(yini, dtype=torch.float64).to(device)
    u_pred = torch.tensor(u_pred, dtype=torch.float64).to(device)
    y_pred = torch.tensor(y_pred, dtype=torch.float64).to(device)
    qp_Q, qp_p, qp_A, qp_b, constant = oracle_to_qp_batch(lambda_g=LAMBDA_G, lambda_y=LAMBDA_Y, upred=u_pred[test_index], ypred=y_pred[test_index], Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, uini=uini[test_index], yini=yini[test_index])
    g_oracle_qpth_tensor = QPFunction(verbose=False, check_Q_spd=False, eps=1e-8)(qp_Q, qp_p, torch.zeros((test_case, 1, n_g), dtype=torch.float64).to(device), torch.ones((test_case, 1), dtype=torch.float64).to(device), qp_A, qp_b).to(device)
    g_oracle_qpth_tensor = g_oracle_qpth_tensor.unsqueeze(-1)
    result_oracle_qpth_tensor = (1 / 2 * g_oracle_qpth_tensor.transpose(-2, -1) @ qp_Q @ g_oracle_qpth_tensor + qp_p.unsqueeze(-1).transpose(-2, -1) @ g_oracle_qpth_tensor + constant).squeeze()
    g_oracle_qpth = g_oracle_qpth_tensor.squeeze(-1).cpu().numpy()
    result_oracle_qpth = result_oracle_qpth_tensor.cpu().numpy()

    print(result_oracle_osqp)
    print(result_oracle_qpth)

    print(f"max diff of result_oracle_osqp and result_oracle_qpth: {np.max(np.abs(result_oracle_osqp - result_oracle_qpth) / np.abs(result_oracle_osqp))}")
    print(f"max diff of g_oracle_osqp and g_oracle_qpth: {np.max(np.abs(g_oracle_osqp - g_oracle_qpth))}")

## compute scores
LAMBDA_G = torch.tensor(LAMBDA_G, dtype=torch.float64).to(device)
LAMBDA_Y = torch.tensor(LAMBDA_Y, dtype=torch.float64).to(device)
Up = torch.tensor(Up, dtype=torch.float64).to(device)
Uf = torch.tensor(Uf, dtype=torch.float64).to(device)
Yp = torch.tensor(Yp, dtype=torch.float64).to(device)
Yf = torch.tensor(Yf, dtype=torch.float64).to(device)
uini = torch.tensor(uini, dtype=torch.float64).to(device)
yini = torch.tensor(yini, dtype=torch.float64).to(device)
u_pred = torch.tensor(u_pred, dtype=torch.float64).to(device)
y_pred = torch.tensor(y_pred, dtype=torch.float64).to(device)

dataset_score = np.zeros(n_traj)
bs = 1000
for i in range(n_traj // bs):
    print(f"COMPUTING SCORE OF {i * bs}/{n_traj}")
    qp_Q, qp_p, qp_A, qp_b, constant = oracle_to_qp_batch(lambda_g=LAMBDA_G, lambda_y=LAMBDA_Y, upred=u_pred[i * bs: (i + 1) * bs], ypred=y_pred[i * bs: (i + 1) * bs], Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, uini=uini[i * bs: (i + 1) * bs], yini=yini[i * bs: (i + 1) * bs])
    g_oracle = QPFunction(verbose=False, check_Q_spd=False, eps=1e-12)(qp_Q, qp_p, torch.zeros((bs, 1, LENGTH - T_INI - HORIZON + 1), dtype=torch.float64).to(device), torch.ones((bs, 1), dtype=torch.float64).to(device), qp_A, qp_b).to(device)
    g_oracle = g_oracle.unsqueeze(-1)
    dataset_score[i * bs: (i + 1) * bs] = (1 / 2 * g_oracle.transpose(-2, -1) @ qp_Q @ g_oracle + qp_p.unsqueeze(-1).transpose(-2, -1) @ g_oracle + constant).squeeze().cpu().numpy()

if n_traj % bs != 0:
    print(f"COMPUTING SCORE OF {n_traj - n_traj % bs}/{n_traj}")
    qp_Q, qp_p, qp_A, qp_b, constant = oracle_to_qp_batch(lambda_g=LAMBDA_G, lambda_y=LAMBDA_Y, upred=u_pred[n_traj - n_traj % bs:], ypred=y_pred[n_traj - n_traj % bs:], Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, uini=uini[n_traj - n_traj % bs:], yini=yini[n_traj - n_traj % bs:])
    g_oracle = QPFunction(verbose=False, check_Q_spd=False, eps=1e-12)(qp_Q, qp_p, torch.zeros((n_traj % bs, 1, LENGTH - T_INI - HORIZON + 1), dtype=torch.float64).to(device), torch.ones((n_traj % bs, 1), dtype=torch.float64).to(device), qp_A, qp_b).to(device)
    g_oracle = g_oracle.unsqueeze(-1)
    dataset_score[n_traj - n_traj % bs:] = (1 / 2 * g_oracle.transpose(-2, -1) @ qp_Q @ g_oracle + qp_p.unsqueeze(-1).transpose(-2, -1) @ g_oracle + constant).squeeze().cpu().numpy()

## plotting
if TUNING:
    np.save(os.path.join(path, 'l2square_score_tuning.npy'), dataset_score)
else:
    if OVERLAP:
        dataset_std = np.load(os.path.join(path, 'traj4learning_std_overlap.npy'))
    else:
        dataset_std = np.load(os.path.join(path, 'traj4learning_std.npy'))
    fig = plt.figure()
    plt.scatter(dataset_std, dataset_score, label="original data", s=0.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Standard deviation')
    plt.ylabel('Score')
    plt.title('Score vs. Standard deviation')
    plt.grid()

    if OVERLAP:
        np.save(os.path.join(path, 'l2square_score_overlap.npy'), dataset_score)
        plt.savefig(os.path.join(path, 'l2square_score_vs_std_overlap.png'))
    else:
        np.save(os.path.join(path, 'l2square_score.npy'), dataset_score)
        plt.savefig(os.path.join(path, 'l2square_score_vs_std.png'))