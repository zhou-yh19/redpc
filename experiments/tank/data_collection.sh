#!/bin/bash

# This script is used to collect data from the tank.
process_std=0.01
measurement_std=0.1

# collect data
python experiments/tank/tank_dataset.py --process-std $process_std --measurement-std $measurement_std --seed 2024 --num-samples 500000

# deepc exp
python experiments/tank/tank_deepc.py --lambda-g1 1.0 --lambda-y1 100 --lambda-g2 100 --lambda-y2 100000 --horizon 20 --norm 12 --process-std $process_std --measurement-std $measurement_std --seed 2024

# data refactor
python experiments/tank/tank_data_refactor.py --process-std $process_std --measurement-std $measurement_std --trajectory-length 30

# data collection
python src/scripts/data_augmentation.py --HORIZON 20 --process-std $process_std --measurement-std $measurement_std --LENGTH 1500 --overlap
python src/scripts/data_augmentation.py --HORIZON 20 --process-std $process_std --measurement-std $measurement_std --LENGTH 1500

# learning objective
python src/scripts/l12_prox.py --HORIZON 20 --LAMBDA-G1 1.0 --LAMBDA-G2 100 --LAMBDA-Y1 100 --LAMBDA-Y2 100000 --process-std $process_std --measurement-std $measurement_std --LENGTH 1500 --overlap
python src/scripts/l12_prox.py --HORIZON 20 --LAMBDA-G1 1.0 --LAMBDA-G2 100 --LAMBDA-Y1 100 --LAMBDA-Y2 100000 --process-std $process_std --measurement-std $measurement_std --LENGTH 1500

# training
python src/grid_train_one_gpu.py