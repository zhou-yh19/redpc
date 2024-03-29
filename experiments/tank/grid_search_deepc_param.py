import subprocess
import itertools
import os
import argparse
file_path = os.path.dirname(__file__)
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--process-std", type=float, default=0.001, help="process noise standard deviation")
parser.add_argument("--measurement-std", type=float, default=0.01, help="measurement noise standard deviation")
args = parser.parse_args()

# Define the path to your Python script
script_path = os.path.join(file_path, "tank_deepc.py")

# Grid search parameters
lambda_g1_values = [0.1, 0.5, 1, 10]  # Example values for lambda_g1
lambda_g2_values = [30, 50, 100]  # Example values for lambda_g2
lambda_y1_values = [0.1, 100, 100000]  # Example values for lambda_y1
lambda_y2_values = [0.1, 100, 100000]  # Example values for lambda_y2
horizon_values = [20, 30, 40]  # Example values for horizon
norm_values = [12]  # Example values for norm
process_std = args.process_std
measurement_std = args.measurement_std

# Function to execute a single instance of the grid search
def execute_command(params):
    lambda_g, lambda_y, lambda_g_, lambda_y_, horizon, norm = params
    command = [
        "python", script_path,
        "--lambda-g1", str(lambda_g),
        "--lambda-y1", str(lambda_y),
        "--lambda-g2", str(lambda_g_),
        "--lambda-y2", str(lambda_y_),
        "--horizon", str(horizon),
        "--norm", str(norm),
        "--process-std", str(process_std),
        "--measurement-std", str(measurement_std),
    ]
    subprocess.run(command)

# Generate all combinations of parameters
all_params = list(itertools.product(lambda_g1_values, lambda_y1_values, lambda_g2_values, lambda_y2_values, horizon_values, norm_values))

# Use ProcessPoolExecutor to run the commands in parallel
with ProcessPoolExecutor(max_workers=12) as executor:
    executor.map(execute_command, all_params)
