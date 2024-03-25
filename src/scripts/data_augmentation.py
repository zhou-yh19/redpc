import torch
import numpy as np
import sys
import os
import argparse
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../../"))

parser = argparse.ArgumentParser()
parser.add_argument("--T-INI", type=int, default=10)
parser.add_argument("--HORIZON", type=int, default=20)
parser.add_argument("--LENGTH", type=int, default=1500)
parser.add_argument("--n-in", type=int, default=2)
parser.add_argument("--n-out", type=int, default=2)
parser.add_argument("--data-size", type=int, default=1000000)
parser.add_argument("--overlap", action="store_true")
parser.add_argument("--system", type=str, default="tank")
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--process-std", type=float, default=0.01)
parser.add_argument("--measurement-std", type=float, default=0.1)

args = parser.parse_args()
T_INI = args.T_INI
HORIZON = args.HORIZON
LENGTH = args.LENGTH
n_in = args.n_in
n_out = args.n_out
data_size = args.data_size
OVERLAP = args.overlap
exp_name = f"process_std_{args.process_std}_measurement_std_{args.measurement_std}"
path = os.path.join(file_path, f"../../experiments/{args.system}/data/{exp_name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_g = LENGTH - (T_INI + HORIZON) + 1
random_seed = args.seed

## data load
if OVERLAP:
    n_traj = n_g
    trajectory = np.load(os.path.join(path, 'trajectory_overlap.npy'))[:n_traj]
    print(f"trajectory_overlap: {trajectory.shape}")
else:
    trajectory = np.load(os.path.join(path, 'trajectory.npy'))
    n_traj = trajectory.shape[0]
    print(f"trajectory: {trajectory.shape}")

## create dataset
'''
The dataset consists of the following three parts apart from the original data:
1. original data with noise of different standard deviation: uini + noise, yini + noise, u_pred + noise, y_pred + noise
2. linear combination of the original data
3. linear combination of the original data with noise of different standard deviation
'''

## trajectory ndarray to tensor
dataset_traj = torch.from_numpy(trajectory).to(device).to(torch.float32)
dataset_std = torch.zeros(n_traj, device=device)
trajectory = torch.from_numpy(trajectory).to(device).to(torch.float32)
torch.manual_seed(random_seed)

batch_size = 10000

for i in range(data_size // batch_size):
    print(f"CREATING DATASET {i * batch_size}/{data_size}")
    batch_point_types = torch.multinomial(torch.tensor([1/3, 1/3, 1/3], device=device), batch_size, replacement=True) + 1
    batch_std = torch.rand(batch_size, device=device) * 10.

    for data_point_type in [1, 2, 3]:
        idxs = (batch_point_types == data_point_type)
        num_idxs = idxs.sum().item()
        if num_idxs == 0:
            continue

        noise_std = batch_std[idxs]

        if data_point_type == 1:
            trajectory_indices = torch.randint(0, n_traj, (num_idxs,), device=device)
            selected_trajectory = trajectory[trajectory_indices] + noise_std.view(-1, 1) * torch.randn((num_idxs, trajectory.shape[1]), device=device)
        elif data_point_type == 2 or data_point_type == 3:
            trajectory_indices = torch.randint(0, n_traj, (num_idxs, 10), device=device)
            trajectory_weights = torch.rand((num_idxs, 10), device=device)
            trajectory_weights = trajectory_weights / trajectory_weights.sum(dim=1, keepdim=True)
            selected_trajectory = torch.bmm(trajectory_weights.unsqueeze(1), trajectory[trajectory_indices]).squeeze(1)

            if data_point_type == 2:
                batch_std[idxs] = torch.zeros(num_idxs, device=device)
            elif data_point_type == 3:
                selected_trajectory += noise_std.view(-1, 1) * torch.randn((num_idxs, trajectory.shape[1]), device=device)
        dataset_traj = torch.cat((dataset_traj, selected_trajectory), dim=0)
        dataset_std = torch.cat((dataset_std, batch_std[idxs]), dim=0)

## save the dataset
dataset_traj = dataset_traj.cpu().numpy()
dataset_std = dataset_std.cpu().numpy()
print(f"dataset_traj: {dataset_traj.shape}")
print(f"dataset_std: {dataset_std.shape}")
if OVERLAP:
    np.save(os.path.join(path, 'traj4learning_overlap.npy'), dataset_traj)
    np.save(os.path.join(path, 'traj4learning_std_overlap.npy'), dataset_std)
else:
    np.save(os.path.join(path, 'traj4learning.npy'), dataset_traj)
    np.save(os.path.join(path, 'traj4learning_std.npy'), dataset_std)