import cvxpy as cp
import numpy as np
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from typing import List, Optional, Callable
from pydeepc import DeePC
from pydeepc.utils import OptimizationProblem, OptimizationProblemVariables

class DeePC_l12(DeePC):
    def build_problem(self,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None,
            lambda_g1: float = 0.,
            lambda_y1: float = 0.,
            lambda_g2: float= 0.,
            lambda_y2: float = 0.) -> OptimizationProblem:
        assert build_loss is not None, "Loss function callback cannot be none"
        assert lambda_g1 >= 0 and lambda_y1 >= 0, "Regularizers must be non-negative"

        self.optimization_problem = False

        # Build variables
        uini = cp.Parameter(shape=(self.M * self.Tini), name='u_ini')
        yini = cp.Parameter(shape=(self.P * self.Tini), name='y_ini')
        u = cp.Variable(shape=(self.M * self.horizon), name='u')
        y = cp.Variable(shape=(self.P * self.horizon), name='y')
        g = cp.Variable(shape=(self.T - self.Tini - self.horizon + 1), name='g')
        slack_y = cp.Variable(shape=(self.Tini * self.P), name='slack_y')
        slack_u = cp.Variable(shape=(self.Tini * self.M), name='slack_u')

        Up, Yp, Uf, Yf = self.Up, self.Yp, self.Uf, self.Yf

        A = np.vstack([Up, Yp, Uf, Yf])
        b = cp.hstack([uini + slack_u, yini + slack_y, u, y])

        # Build constraints
        constraints = [A @ g == b]

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
        _regularizers = lambda_g1 * cp.norm(g, p=1) if lambda_g1 > DeePC._SMALL_NUMBER else 0
        _regularizers += lambda_g2 * cp.quad_form(g, np.eye(self.T - self.Tini - self.horizon + 1)) if lambda_g2 > DeePC._SMALL_NUMBER else 0
        _regularizers += lambda_y1 * cp.norm(slack_y, p=1) if lambda_y1 > DeePC._SMALL_NUMBER else 0
        _regularizers += lambda_y2 * cp.quad_form(slack_y, np.eye(self.Tini * self.P)) if lambda_y2 > DeePC._SMALL_NUMBER else 0

        problem_loss = _loss + _regularizers

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

        return self.optimization_problem