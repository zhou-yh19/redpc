import cvxpy as cp
import numpy as np
from pydeepc import DeePC
from pydeepc.utils import OptimizationProblem, OptimizationProblemVariables
from .deepc_qp import deepc_to_qp_cvxpy

class DeePC_QP(DeePC):
    def build_problem(self,
            Q: np.ndarray = None, R: np.ndarray = None, ref: np.ndarray = None,
            u_upper: np.ndarray = None, u_lower: np.ndarray = None,
            y_upper: np.ndarray = None, y_lower: np.ndarray = None,
            lambda_g: float = 0.,
            lambda_y: float = 0.) -> OptimizationProblem:
        """
        Builds the DeePC optimization problem with l2 square regularization as QP:
        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :param lambda_g:            non-negative scalar. Regularization factor for g. Used for
                                    stochastic/non-linear systems.
        :param lambda_y:            non-negative scalar. Regularization factor for y_init. Used for
                                    stochastic/non-linear systems.
        :return:                    Parameters of the optimization problem
        """
        # assert Q and R and ref is not None, "Q, R and ref must be provided"
        # assert u_upper and u_lower and y_upper and y_lower is not None, "Bounds must be provided"
        assert lambda_g > 0 and lambda_y > 0, "Regularizers must be positive"

        self.optimization_problem = False

        uini = cp.Parameter(shape=(self.M * self.Tini), name='u_ini')
        yini = cp.Parameter(shape=(self.P * self.Tini), name='y_ini')
        u = cp.Variable(shape=(self.M * self.horizon), name='u')
        y = cp.Variable(shape=(self.P * self.horizon), name='y')
        g = cp.Variable(shape=(self.T - self.Tini - self.horizon + 1), name='g')
        slack_y = cp.Variable(shape=(self.Tini * self.P), name='slack_y')
        slack_u = cp.Variable(shape=(self.Tini * self.M), name='slack_u')

        Up, Yp, Uf, Yf = self.Up, self.Yp, self.Uf, self.Yf
        
        qp_Q, qp_p, qp_G, qp_h, qp_A, qp_b, constant = deepc_to_qp_cvxpy(lambda_g, lambda_y, Q, R, ref, Up, Uf, Yp, Yf, uini, yini, u_upper, u_lower, y_upper, y_lower)
        
        constraints = [qp_A @ g == qp_b, qp_G @ g <= qp_h]

        # problem_loss = _loss + _regularizers
        problem_loss = 1 / 2 * cp.quad_form(g, qp_Q) + qp_p @ g

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(
                u_ini = uini, y_ini = yini, u = u, y = y, g = g, slack_y = slack_y, slack_u = slack_u),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        self.constant = constant

        return self.optimization_problem