import subprocess
from multiprocessing import Pool
import itertools
from concurrent.futures import ProcessPoolExecutor

def cmd_train(lr, m, n, iters, huber_delta, seed, overlap, exp_name, epoch, trajectory_length, trajectory_size, horizon, norm, approximator, process_std, measurement_std):
    """Construct the command to run the training process."""
    if overlap:
        return f"CUDA_VISIBLE_DEVICES=0 python src/train.py --lr {lr} --m {m} --n {n} --iters {iters} --huber_delta {huber_delta} --seed {seed} --overlap --exp_name {exp_name} --epochs {epoch} --trajectory_length {trajectory_length} --norm {norm} --approximator {approximator}"
    else:
        return f"CUDA_VISIBLE_DEVICES=0 python src/train.py --lr {lr} --m {m} --n {n} --iters {iters} --huber_delta {huber_delta} --seed {seed} --exp_name {exp_name} --epochs {epoch} --trajectory_length {trajectory_length} --norm {norm} --approximator {approximator}"

def cmd_test_redpc(lr, m, n, iters, huber_delta, seed, overlap, exp_name, epoch, trajectory_length, trajectory_size, horizon, norm, approximator, process_std, measurement_std):
    if overlap:
        return f"python experiments/tank/tank_redpc.py --lr {lr} --m {m} --n {n} --iters {iters} --huber_delta {huber_delta} --seed {seed} --overlap --epochs {epoch} --trajectory_size {trajectory_size} --horizon {horizon} --norm {norm} --approximator {approximator} --process_std {process_std} --measurement_std {measurement_std}"
    else:
        return f"python experiments/tank/tank_redpc.py --lr {lr} --m {m} --n {n} --iters {iters} --huber_delta {huber_delta} --seed {seed} --epochs {epoch} --trajectory_size {trajectory_size} --horizon {horizon} --norm {norm} --approximator {approximator} --process_std {process_std} --measurement_std {measurement_std}"

def execute(cmd1, cmd2):
    """Execute a single training command."""
    print(f"Executing: {cmd1}")
    subprocess.run(cmd1, shell=True)
    print(f"Executing: {cmd2}")
    subprocess.run(cmd2, shell=True)


if __name__ == "__main__":
    # Define hyperparameters
    lrs = [0.1]
    ms = [20, 30, 40]
    ns = [50, 60, 80, 100]
    iters = [20 ,50, 100]
    huber_deltas = [10000]
    seeds = [2022]
    epochs = [200]
    overlap = [True]
    exp_name = ["process_std_0.01_measurement_std_0.1"]
    process_std = [0.01]
    measurement_std = [0.1]
    horizon = [20]
    trajectory_length = [30]
    trajectory_size = [1500]
    norm = [0]
    approximator = ["L12Prox"]

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(lrs, ms, ns, iters, huber_deltas, seeds, overlap, exp_name, epochs, trajectory_length, trajectory_size, horizon, norm, approximator, process_std, measurement_std))
    
    # Convert hyperparameters to commands
    commands_training = [cmd_train(*params) for params in hyperparameter_combinations]
    commands_testing = [cmd_test_redpc(*params) for params in hyperparameter_combinations]

    # Number of processes should be limited based on your GPU's capability
    # and the memory requirement of each task.
    # Adjust this number based on your system's specifications.
    # use gpu to do the training, and use cpu to do the testing
    # test after training
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(execute, commands_training, commands_testing)
