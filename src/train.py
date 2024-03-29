# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import argparse
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../"))
from torch import nn, optim
from src.modules.deepc_score_approximator import DeepCScoreApproximator
from src.utils.dataloader import create_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=10000, help='batch size')
parser.add_argument('--m', type=int, default=48, help='number of constraints')
parser.add_argument('--n', type=int, default=2, help='number of variables')
parser.add_argument('--approximator', type=str, default='QP', help='approximator name')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument("--iters", type=int, default=20, help="number of iterations of solver")
parser.add_argument("--mlp-size-last", type=int, default=64)
parser.add_argument("--n-in", type=int, default=2)
parser.add_argument("--n-out", type=int, default=2)
parser.add_argument("--trajectory-length", type=int, default=50)
parser.add_argument("--batch-num", type=int, default=10)
parser.add_argument("--overlap", action="store_true", help="Enable overlap feature (default: %(default)s)")
parser.add_argument("--system", type=str, default='tank', help="system name")
parser.add_argument("--process-std", type=float, default=0.01)
parser.add_argument("--measurement-std", type=float, default=0.1)
parser.add_argument("--norm", type=int, default=2, help="norm of the score")

args = parser.parse_args()
LR = args.lr
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
batch_num = args.batch_num
approximator_name = args.approximator
if approximator_name == 'QP' or 'L1' or 'L12' or 'L12Prox':
    approximation_size = [args.m, args.n]
elif approximator_name == 'MLP':
    approximation_size = [args.mlp_size_last * i for i in (4, 2, 1)] 
device = args.device
random_seed = args.seed
iters = args.iters
n_in = args.n_in
n_out = args.n_out
trajectory_length = args.trajectory_length
process_std = args.process_std
measurement_std = args.measurement_std
exp_name = f"process_std_{process_std}_measurement_std_{measurement_std}"
overlap = args.overlap
norm = args.norm
np.random.seed(random_seed)
torch.manual_seed(random_seed)

path = os.path.join(file_path, f"../experiments/{args.system}")
exp_path = os.path.join(path, f'runs/{exp_name}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
case_name = f"{approximator_name}_iters_{iters}_nqp_{args.n}_mqp_{args.m}_lr_{LR}_seed_{random_seed}_overlap_{overlap}_norm_{norm}"
case_path = os.path.join(exp_path, case_name)
print(f"case_path: {case_path}")
if not os.path.exists(case_path):
    os.makedirs(case_path)
data_path = os.path.join(path, f"data/{exp_name}")

## load data
dataloaders = create_dataloader(data_path, norm=norm, batch_size=BATCH_SIZE, device=device, overlap=overlap)

## define the model
model = DeepCScoreApproximator(n_in=n_in, n_out=n_out, N=trajectory_length, approximator_name=approximator_name, approximator_size=approximation_size, device=device, iters=iters)
print(model)

## optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# # loss function
criterion = nn.MSELoss(reduction="mean")
# criterion = nn.L1Loss(reduction="mean")

## record the loss and error to draw the curve
score_loss = {}
score_loss['train'] = []
score_loss['val'] = []
traj_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(111, title='loss')

def draw_curve(fig, ax, filename, current_epoch):
    traj_epoch.append(current_epoch)
    ax.plot(traj_epoch, score_loss['train'], 'b-', label='train')
    ax.plot(traj_epoch, score_loss['val'], 'r-', label='val')
    ax.set_yscale('log')
    if current_epoch == 0:
        ax.legend()
    fig.savefig(filename)

for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
    print('-' * 10)

    ## Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0 # loss of each phase
        count = 0
        # Iterate over data.
        for data in dataloaders[phase]:
            if count > batch_num:
                break

            count = count + 1

            inputs, labels = data
            now_batch_size, n = inputs.shape
            if now_batch_size < BATCH_SIZE:  # skip the last batch
                continue

            # zero the parameter gradients
            optimizer.zero_grad()

            # -------- forward --------
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels[:, :-1])
            del inputs

            # -------- backward + optimize --------
            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            del loss

        epoch_loss = running_loss / count

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        score_loss[phase].append(epoch_loss)

        # deep copy the model
        if phase == 'val':
            draw_curve(fig, ax0, os.path.join(case_path, 'loss'), epoch)

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(case_path, 'checkpoint_{}.pth'.format(epoch)))

## test model of validation set
if norm == 0:
    model.train(False)
    fig_test, ax = plt.subplots()
    for data in dataloaders['val']:
        inputs, labels = data
        outputs = model(inputs).squeeze()
        ax.scatter(labels[:, -1].detach().cpu().numpy(), np.sum((outputs.detach().cpu().numpy() - labels[:, :-1].detach().cpu().numpy()) ** 2, axis=1), c='r', s=0.1)
    ax.legend(["Prediction", "Ground Truth"])
    ax.set_xlabel('Standard deviation')
    ax.set_ylabel('Score')
    ax.set_title('Score vs. standard deviation')
    ax.grid()
    fig_test.savefig(os.path.join(case_path, 'test.png'))
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig_test.savefig(os.path.join(case_path, 'test_log.png'))
else:
    model.train(False)
    fig_test, ax = plt.subplots()
    for data in dataloaders['val']:
        inputs, labels = data
        outputs = model(inputs).squeeze()
        ax.scatter(labels[:, 1].detach().cpu().numpy(), outputs.detach().cpu().numpy(), c='r', s=0.1)
        ax.scatter(labels[:, 1].detach().cpu().numpy(), labels[:, 0].detach().cpu().numpy(), c='b', s=0.1)
    ax.legend(["Prediction", "Ground Truth"])
    ax.set_xlabel('Standard deviation')
    ax.set_ylabel('Score')
    ax.set_title('Score vs. standard deviation')
    ax.grid()
    fig_test.savefig(os.path.join(case_path, 'test.png'))
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig_test.savefig(os.path.join(case_path, 'test_log.png'))
    
