import subprocess
from multiprocessing import Pool
import itertools
from concurrent.futures import ProcessPoolExecutor

def cmd_train(lr, m, n, iters, seed, overlap, epoch, trajectory_length, trajectory_size, horizon, norm, approximator, process_std, measurement_std):
    """Construct the command to run the training process."""
    if overlap:
        return f"CUDA_VISIBLE_DEVICES=0 python src/train.py --lr {lr} --m {m} --n {n} --iters {iters} --seed {seed} --overlap --epochs {epoch} --trajectory-length {trajectory_length} --norm {norm} --approximator {approximator} --process-std {process_std} --measurement-std {measurement_std}"
    else:
        return f"CUDA_VISIBLE_DEVICES=0 python src/train.py --lr {lr} --m {m} --n {n} --iters {iters} --seed {seed}  --epochs {epoch} --trajectory-length {trajectory_length} --norm {norm} --approximator {approximator} --process-std {process_std} --measurement-std {measurement_std}"

def cmd_test_redpc(lr, m, n, iters, seed, overlap, epoch, trajectory_length, trajectory_size, horizon, norm, approximator, process_std, measurement_std):
    if overlap:
        return f"python experiments/tank/tank_redpc.py --lr {lr} --m {m} --n {n} --iters {iters} --training-seed {seed} --overlap --epochs {epoch} --trajectory-size {trajectory_size} --horizon {horizon} --norm {norm} --approximator {approximator} --process-std {process_std} --measurement-std {measurement_std}"
    else:
        return f"python experiments/tank/tank_redpc.py --lr {lr} --m {m} --n {n} --iters {iters} --training-seed {seed}  --epochs {epoch} --trajectory-size {trajectory_size} --horizon {horizon} --norm {norm} --approximator {approximator} --process-std {process_std} --measurement-std {measurement_std}"

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
    seeds = [2022]
    epochs = [10]
    overlap = [True]
    process_std = [0.01]
    measurement_std = [0.1]
    horizon = [20]
    trajectory_length = [30]
    trajectory_size = [1500]
    norm = [0] # We abuse the norm parameter to specify the type of learning objective
    approximator = ["L12Prox"]

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(lrs, ms, ns, iters, seeds, overlap, epochs, trajectory_length, trajectory_size, horizon, norm, approximator, process_std, measurement_std))
    
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
