# REDPC: A Real-time Efficient approximate DeePC

## Python version

The code is tested with Python 3.10

## Installation
```
pip install -r requirements.txt
git clone git@github.com:zhou-yh19/PyDeePC.git
cd PyDeePC
pip install -e .
```

## Code structure

- `src/modules`: PyTorch modules for approximating the scoring function
- `src/scripts`: Scripts for data augmentation, and computing different scoring function
- `src/utils`: Utility functions (customized PyTorch operations, tensor operations, etc.)
- `src/train.py`: Training script for the scoring function
- `src/grid_train_one_gpu.py`: Training script for the scoring function on a single GPU
- `experiments`: Sample scripts for running experiments
- `test`: julia scripts for testing solver

## License

The project is released under the MIT license. See [LICENSE](LICENSE) for details.

Part of the project is modified from [PyDeePC](https://github.com/rssalessio/PyDeePC)
