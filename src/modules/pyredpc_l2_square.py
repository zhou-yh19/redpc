import cvxpy as cp
import numpy as np
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from typing import Dict, Union, Optional, List, Callable, Tuple
from pydeepc import DeePC
from pydeepc.utils import OptimizationProblem, OptimizationProblemVariables, Data

class RedPC(DeePC):
    def __init__(self, approximator_name: str, param: Dict[str, Union[int, float, np.ndarray]], Tini: int, horizon: int):
        """
        min QR loss + learnable regularization
        s.t. Affine constraints
        :param approximator_name:   Name of the approximator to be used. It can be one of the following: QP, l1.
        :param param:               Dictionary with the parameters of the approximator.
        :param Tini:                number of samples needed to estimate initial conditions
        :param horizon:             horizon length
        """
        self.approximator_name = approximator_name
        self.Tini = Tini
        self.horizon = horizon
        self.M = param['n_u']
        self.P = param['n_y']
        self.G = param['G']
        self.H = param['H']
        self.dim = param['n']
        if approximator_name == 'QP':
            self.Pinv = param['Pinv']

        elif approximator_name == 'L12' or approximator_name == 'L12Prox':
            self.D1 = param['D1']
            self.D2 = param['D2']

        self.optimization_problem = None

    def build_problem(self,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None
            ) -> OptimizationProblem:

        assert build_loss is not None, "Loss function callback cannot be none"
        self.optimization_problem = False

        # Build variables
        uini = cp.Parameter(shape=(self.M * self.Tini), name='u_ini')
        yini = cp.Parameter(shape=(self.P * self.Tini), name='y_ini')
        u = cp.Variable(shape=(self.M * self.horizon), name='u')
        y = cp.Variable(shape=(self.P * self.horizon), name='y')
        g = cp.Variable(shape=(self.dim), name='g')
        slack_y = cp.Variable(shape=(self.Tini * self.P), name='slack_y')
        slack_u = cp.Variable(shape=(self.Tini * self.M), name='slack_u')

        T = cp.hstack([uini, u, yini, y])

        # Build constraints
        if self.approximator_name == 'QP':
            constraints = [self.G @ g + self.H @ T >= 0]
        elif self.approximator_name == 'L12' or self.approximator_name == 'L12Prox':
            constraints = [self.G @ g == self.H @ T]

        u = cp.reshape(u, (self.horizon, self.M), order='C')
        y = cp.reshape(y, (self.horizon, self.P), order='C')

        _constraints = build_constraints(u, y) if build_constraints is not None else (None, None)

        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)

        # Build loss
        _loss = build_loss(u, y)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        # Add regularizers
        if self.approximator_name == 'QP':
            print(f"self.Pinv: {self.Pinv.shape}")
            print(f"np.linalg.inv(self.Pinv): {np.linalg.inv(self.Pinv).shape}")
            _regularizers = 1 / 2 * cp.quad_form(g, np.linalg.inv(self.Pinv))
        elif self.approximator_name == 'L12' or self.approximator_name == 'L12Prox':
            _regularizers = cp.norm(cp.norm(cp.multiply(self.D1, g), 1)) + cp.quad_form(g, np.diag(self.D2 ** 2))
        problem_loss = _loss + _regularizers

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the RedPC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(
                u_ini = uini, y_ini = yini, u = u, y = y, g = g, slack_y = slack_y, slack_u = slack_u),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        return self.optimization_problem

    def solve(
            self,
            data_ini: Data,
            **cvxpy_kwargs
        ) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray, OptimizationProblemVariables]]]:
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data_ini:            A tuple of input/output data used to estimate initial condition.
                                    Data should have shape Tini x M where Tini is the batch size and
                                    M is the number of features
        :param cvxpy_kwargs:        All arguments that need to be passed to the cvxpy solve method.
        :return u_optimal:          Optimal input signal to be applied to the system, of length `horizon`
        :return info:               A dictionary with 5 keys:
                                    info['variables']: variables of the optimization problem
                                    info['value']: value of the optimization problem
                                    info['u_optimal']: the same as the first value returned by this function
        """
        assert len(data_ini.u.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data_ini.y.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data_ini.u.shape[1] == self.M, "Incorrect number of features for the input signal"
        assert data_ini.y.shape[1] == self.P, "Incorrect number of features for the output signal"
        assert data_ini.y.shape[0] == data_ini.u.shape[0], "Input/output data must have the same length"
        assert data_ini.y.shape[0] == self.Tini, f"Invalid size"
        assert self.optimization_problem is not None, "Problem was not built"


        # Need to transpose to make sure that time is over the columns, and features over the rows
        uini, yini = data_ini.u[:self.Tini].flatten(), data_ini.y[:self.Tini].flatten()

        self.optimization_problem.variables.u_ini.value = uini
        self.optimization_problem.variables.y_ini.value = yini

        try:
            result = self.optimization_problem.problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            raise Exception('Problem is unbounded')

        u_optimal = self.optimization_problem.variables.u.value.reshape(self.horizon, self.M)
        info = {
            'value': result, 
            'variables': self.optimization_problem.variables,
            'u_optimal': u_optimal
            }

        return u_optimal, info
