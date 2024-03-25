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
python src/scripts/data_augmentation_offline.py --HORIZON 20 --exp_name process_std_0.01_measurement_std_0.1 --LENGTH 1500 --overlap
python src/scripts/data_augmentation_offline.py --HORIZON 20 --exp_name process_std_0.01_measurement_std_0.1 --LENGTH 1500

# score
python src/scripts/l12_prox.py --HORIZON 20 --LAMBDA_G 1.0 --LAMBDA_Y 100 --exp_name process_std_0.01_measurement_std_0.1 --LAMBDA_G_ 100 --LAMBDA_Y_ 100000 --LENGTH 1500 --overlap

# train
python src/scripts/l12_score.py --HORIZON 20 --LAMBDA_G 0.1 --LAMBDA_Y 100000 --exp_name process_std_0.01_measurement_std_0.1 --LAMBDA_G_ 30 --LAMBDA_Y_ 100000 --LENGTH 1500 --overlap
python src/grid_train.py