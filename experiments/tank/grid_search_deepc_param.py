import subprocess
import itertools
import os
import argparse
file_path = os.path.dirname(__file__)
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--process_std", type=float, default=0.001, help="process noise standard deviation")
parser.add_argument("--measurement_std", type=float, default=0.01, help="measurement noise standard deviation")
args = parser.parse_args()

# Define the path to your Python script
script_path = os.path.join(file_path, "tank_deepc.py")

# Grid search parameters
lambda_g_values = [0.1, 0.5, 1, 10]  # Example values for lambda_g
lambda_g_values_ = [30, 50, 100]  # Example values for lambda_g_
lambda_y_values = [0.1, 100, 100000]  # Example values for lambda_y
lambda_y_values_ = [0.1, 100, 100000]  # Example values for lambda_y_
horizon_values = [20, 30, 40]  # Example values for horizon
norm_values = [12]  # Example values for norm
process_std = args.process_std
measurement_std = args.measurement_std

# Function to execute a single instance of the grid search
def execute_command(params):
    lambda_g, lambda_y, lambda_g_, lambda_y_, horizon, norm = params
    command = [
        "python", script_path,
        "--lambda_g", str(lambda_g),
        "--lambda_y", str(lambda_y),
        "--lambda_g_", str(lambda_g_),
        "--lambda_y_", str(lambda_y_),
        "--horizon", str(horizon),
        "--norm", str(norm),
        "--process_std", str(process_std),
        "--measurement_std", str(measurement_std),
    ]
    subprocess.run(command)

# Generate all combinations of parameters
all_params = list(itertools.product(lambda_g_values, lambda_y_values, lambda_g_values_, lambda_y_values_, horizon_values, norm_values))

# Use ProcessPoolExecutor to run the commands in parallel
with ProcessPoolExecutor(max_workers=12) as executor:
    executor.map(execute_command, all_params)
