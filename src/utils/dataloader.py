import torch
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(
    path,
    norm=2,
    batch_size=1024,
    overlap=False,
    shuffle=True,
    device='cuda'
    ):

    if norm == 1:
        score_name = "l1"
    elif norm == 2:
        score_name = "l2square"
    elif norm == 12:
        score_name = "l12"
    elif norm == 0:
        score_name = "l12_prox_tau"
    ## load data
    if overlap:
        trajectory = np.load(os.path.join(path, "traj4learning_overlap.npy"))
        if norm == 0:
            score = np.load(os.path.join(path, f"{score_name}_score_overlap.npy")).reshape(-1, 120)
        else:
            score = np.load(os.path.join(path, f"{score_name}_score_overlap.npy")).reshape(-1, 1)
        std = np.load(os.path.join(path, "traj4learning_std_overlap.npy")).reshape(-1, 1)
        score = np.concatenate([score, std], axis=1)
    else:
        trajectory = np.load(os.path.join(path, "traj4learning.npy"))
        score = np.load(os.path.join(path, f"{score_name}_score.npy")).reshape(-1, 1)
        std = np.load(os.path.join(path, "traj4learning_std.npy")).reshape(-1, 1)
        score = np.concatenate([score, std], axis=1)

    ## ndarray to tensor
    trajectory = torch.tensor(trajectory, dtype=torch.float32).to(device)
    score = torch.tensor(score, dtype=torch.float32).to(device)
    print(f"trajectory shape: {trajectory.shape}")
    print(f"score shape: {score.shape}")

    ## dataloader construction
    traj_train, traj_val, score_train, score_val = train_test_split(trajectory, score, test_size=0.2)
    train_dataloader = DataLoader(TensorDataset(traj_train, score_train), batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(TensorDataset(traj_val, score_val), batch_size=batch_size, shuffle=shuffle)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Report split sizes
    print('Training set has {} instances'.format(len(train_dataloader.dataset)))
    print('Validation set has {} instances'.format(len(val_dataloader.dataset)))

    return dataloaders

if __name__ == '__main__':
    ## test code
    path = Path("../data")
    dataloaders = create_dataloader(path)