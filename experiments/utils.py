# The code in this file is adapted from https://github.com/rssalessio/PyDeePC
import numpy as np
import scipy.signal as scipysig
from typing import Optional, NamedTuple, Tuple

class Data(NamedTuple):
    """
    Tuple that contains input/output data
    :param u: input data
    :param y: output data
    """
    u: np.ndarray
    y: np.ndarray

class System(object):
    """
    Represents a dynamical system that can be simulated
    """
    def __init__(self, sys: scipysig.StateSpace, x0: Optional[np.ndarray] = None):
        """
        :param sys: a linear system
        :param x0: initial state
        """
        assert x0 is None or sys.A.shape[0] == len(x0), 'Invalid initial condition'
        self.sys = sys
        self.x0 = x0 if x0 is not None else np.zeros(sys.A.shape[0])
        self.u = None
        self.y = None

    def get_last_n_samples(self, n: int) -> Data:
        """
        Returns the last n samples
        :param n: integer value
        """
        assert self.u.shape[0] >= n, 'Not enough samples are available'
        return Data(self.u[-n:], self.y[-n:])

    def get_all_samples(self) -> Data:
        """
        Returns all samples
        """
        return Data(self.u, self.y)

    def reset(self, data_ini: Optional[Data] = None, x0: Optional[np.ndarray] = None):
        """
        Reset initial state and collected data
        """
        self.u = None if data_ini is None else data_ini.u
        self.y = None if data_ini is None else data_ini.y
        self.x0 = x0 if x0 is not None else np.zeros(self.sys.A.shape[0])

    def simulate(self, u: np.ndarray, process_std: float = 0.1, measurement_std: float = 0.1) -> Data:
        """
        Simulate the system with given input signals
        :param u: input signal
        :param process_std: standard deviation of the process noise
        :param measurement_std: standard deviation of the measurement noise
        :return: tuple that contains the (input,output) of the system
        """
        T = len(u)
        y = np.zeros((T, self.sys.C.shape[0]))
        for t in range(T):
            y[t] = self.sys.C @ self.x0 + np.random.normal(0, measurement_std, size = self.sys.C.shape[0])
            self.x0 = self.sys.A @ self.x0 + self.sys.B @ u[t] + np.random.normal(0, process_std, size = self.x0.shape)
        
        self.u = np.vstack([self.u, u]) if self.u is not None else u
        self.y = np.vstack([self.y, y]) if self.y is not None else y

        return Data(u, y)

def controllability_matrix(A, B):
    """
    Computes the controllability matrix of a linear system and checks if the system is controllable
    :param A: state matrix
    :param B: input matrix
    :return: controllability matrix and a boolean value that indicates if the system is controllable
    """
    n = A.shape[0]  # Number of states
    C = B  # Initialize the controllability matrix with B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    C_matrix = C
    is_controllable = np.linalg.matrix_rank(C_matrix) == A.shape[0]
    return C_matrix, is_controllable

# Compute the observability matrix
def observability_matrix(A, C):
    """
    Computes the observability matrix of a linear system and checks if the system is observable
    :param A: state matrix
    :param C: output matrix
    """
    n = A.shape[0]  # Number of states
    O = C  # Initialize the observability matrix with C
    for i in range(1, n):
        O = np.vstack((O, C @ np.linalg.matrix_power(A, i)))
    O_matrix = O
    is_observable = np.linalg.matrix_rank(O_matrix) == A.shape[0]
    return O_matrix, is_observable

def create_hankel_matrix(data: np.ndarray, order: int) -> np.ndarray:
    """
    Create an Hankel matrix of order L from a given matrix of size TxM,
    where M is the number of features and T is the batch size.
    Note that we need L <= T.
    :param data:    A matrix of data (size TxM). 
                    T is the batch size and M is the number of features
    :param order:   the order of the Hankel matrix (L)
    :return:        The Hankel matrix of type np.ndarray
    """
    data = np.array(data)
    
    assert len(data.shape) == 2, "Data needs to be a matrix"

    T,M = data.shape
    assert T >= order and order > 0, "The number of data points needs to be larger than the order"

    H = np.zeros((order * M, (T - order + 1)))
    for idx in range (T - order + 1):
        H[:, idx] = data[idx:idx+order, :].flatten()
    return H

def split_data(data: Data, Tini: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Utility function used to split the data into past data and future data.
    Constructs the Hankel matrix for the input/output data, and uses the first
    Tini rows to create the past data, and the last 'horizon' rows to create
    the future data.
    For more info check eq. (4) in https://arxiv.org/pdf/1811.05890.pdf
    :param data:    A tuple of input/output data. Data should have shape TxM
                    where T is the batch size and M is the number of features
    :param Tini:    number of samples needed to estimate initial conditions
    :param horizon: horizon
    :return:        Returns Up,Uf,Yp,Yf (see eq. (4) of the original DeePC paper)
    """
    assert Tini >= 1, "Tini cannot be lower than 1"
    assert horizon >= 1, "Horizon cannot be lower than 1"

    Mu, My = data.u.shape[1], data.y.shape[1]
    Hu = create_hankel_matrix(data.u, Tini + horizon)
    Hy = create_hankel_matrix(data.y, Tini + horizon)

    Up, Uf = Hu[:Tini * Mu], Hu[-horizon * Mu:]
    Yp, Yf = Hy[:Tini * My], Hy[-horizon * My:]
    
    print(f"Up: {Up.shape}, Uf: {Uf.shape}, Yp: {Yp.shape}, Yf: {Yf.shape}")
    HL = np.concatenate((Up, Yp, Uf, Yf), axis=0)
    ## rank of the Hankel matrix
    print(f"Rank of Hankel matrix: {np.linalg.matrix_rank(HL)}")

    return Up, Uf, Yp, Yf