# %%
import math
import numpy as np
import torch
import cvxpy as cp

def deepc_to_qp(lambda_g: float = 0., lambda_y: float = 0., Q: np.ndarray = None, R: np.ndarray = None, ref: np.ndarray = None,
                Up: np.ndarray = None, Uf: np.ndarray = None, Yp: np.ndarray = None, Yf: np.ndarray = None, uini: np.ndarray = None, yini: np.ndarray = None,
                u_upper: np.ndarray = None, u_lower: np.ndarray = None, y_upper: np.ndarray = None, y_lower: np.ndarray = None):
    """
    Convert DeePC problem to QP problem
    """
    assert lambda_g >= 0 and lambda_y >= 0, "Regularizers must be non-negative"
    if math.isclose(lambda_g, 0.) and math.isclose(lambda_y, 0.):
        ## TODO: handle this case
        print("No regularizer is provided")
        return None

    Q = np.kron(np.eye(Yf.shape[0] // y_upper.shape[0]), Q)
    R = np.kron(np.eye(Uf.shape[0] // u_upper.shape[0]), R)
    qp_Q = 2 * (Yf.T @ Q @ Yf + Uf.T @ R @ Uf + lambda_y * Yp.T @ Yp + lambda_g * np.eye(Uf.shape[1]))
    qp_p = - 2  * (Yf.T @ Q @ ref + lambda_y * Yp.T @ yini)
    qp_A = Up
    qp_b = uini
    qp_G = np.vstack([Uf, -Uf, Yf, -Yf])
    qp_h = np.concatenate([np.tile(u_upper, (Uf.shape[0] // u_upper.shape[0])), -np.tile(u_lower, (Uf.shape[0] // u_lower.shape[0])), np.tile(y_upper, (Yf.shape[0] // y_upper.shape[0])), -np.tile(y_lower, (Yf.shape[0] // y_lower.shape[0]))])
    # g = QPFunction(verbose=False)(qp_Q, qp_p, qp_G, qp_h, qp_A, qp_b)
    constant = ref.T @ Q @ ref + lambda_y * yini.T @ yini
    return qp_Q, qp_p, qp_G, qp_h, qp_A, qp_b, constant

def oracle_to_qp(lambda_g: float = 0., lambda_y: float = 0., upred: np.ndarray = None, ypred: np.ndarray = None,
                Up: np.ndarray = None, Uf: np.ndarray = None, Yp: np.ndarray = None, Yf: np.ndarray = None, uini: np.ndarray = None, yini: np.ndarray = None):
    """
    Convert Oracle problem to QP problem
    """
    assert lambda_g >= 0 and lambda_y >= 0, "Regularizers must be non-negative"
    if math.isclose(lambda_g, 0.) and math.isclose(lambda_y, 0.):
        ## TODO: handle this case
        print("No regularizer is provided")
        return None

    qp_Q = 2 * (lambda_g * np.eye(Uf.shape[1]) + lambda_y * Yp.T @ Yp)
    qp_p = - 2  * (lambda_y * Yp.T @ yini)
    qp_A = np.vstack([Up, Uf, Yf])
    qp_b = np.concatenate([uini, upred, ypred])
    constant = lambda_y * yini.T @ yini
    return qp_Q, qp_p, qp_A, qp_b, constant

## torch version of function deepc_to_qp
def deepc_to_qp_batch(lambda_g: torch.Tensor, lambda_y: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, ref: torch.Tensor,
                      Up: torch.Tensor, Uf: torch.Tensor, Yp: torch.Tensor, Yf: torch.Tensor, uini: torch.Tensor, yini: torch.Tensor,
                      u_upper: torch.Tensor, u_lower: torch.Tensor, y_upper: torch.Tensor, y_lower: torch.Tensor, device: torch.device = torch.device("cuda")):
    """
    Convert batch of DeePC problems to QP problems
    """
    B = uini.shape[0]  # Assuming Uf has shape [B, T, n_u] where B is batch size
    assert lambda_g.ge(0).all() and lambda_y.ge(0).all(), "Regularizers must be non-negative for all batch elements"

    # Handle the case where no regularizer is provided for all batch elements
    if (lambda_g.eq(0) & lambda_y.eq(0)).all():
        print("No regularizer is provided for all batch elements")
        return None
    
    lambda_g = lambda_g.to(device)
    lambda_y = lambda_y.to(device)
    ref = ref.reshape(B, -1, 1).to(device)
    Up = Up.unsqueeze(0).repeat(B, 1, 1).to(device)
    Yp = Yp.unsqueeze(0).repeat(B, 1, 1).to(device)
    Uf = Uf.unsqueeze(0).repeat(B, 1, 1).to(device)
    Yf = Yf.unsqueeze(0).repeat(B, 1, 1).to(device)
    uini = uini.reshape(B, -1, 1).to(device)
    yini = yini.reshape(B, -1, 1).to(device)
    u_upper = u_upper.to(device)
    u_lower = u_lower.to(device)
    y_upper = y_upper.to(device)
    y_lower = y_lower.to(device)

    # Assuming Q and R are given as [B, n, n] and need to be expanded for each batch element
    eye_y = torch.eye(Yf.shape[1] // y_upper.shape[0]).repeat(B, 1, 1)  # Adjusting for batch dimension
    Q_expanded = torch.kron(eye_y, Q).to(device)  # This may require a custom batch-wise Kronecker product implementation
    eye_u = torch.eye(Uf.shape[1] // u_upper.shape[0]).repeat(B, 1, 1)
    R_expanded = torch.kron(eye_u, R).to(device)  # Similarly, adjust for batch-wise operation
    
    # Calculating qp_Q, qp_p, qp_G, qp_h, qp_A, qp_b with batch support
    # RuntimeError: expected scalar type Float but found Double
    qp_Q = 2 * (Yf.transpose(-2, -1) @ Q_expanded @ Yf + Uf.transpose(-2, -1) @ R_expanded @ Uf +
                lambda_y.unsqueeze(-1).unsqueeze(-1) * Yp.transpose(-2, -1) @ Yp +
                lambda_g.unsqueeze(-1).unsqueeze(-1) * torch.eye(Uf.shape[-1]).repeat(B, 1, 1).to(device))
    qp_p = -2 * (Yf.transpose(-2, -1) @ Q_expanded @ ref + lambda_y.unsqueeze(-1).unsqueeze(-1) * Yp.transpose(-2, -1) @ yini).reshape(B, -1)
    qp_A = Up
    qp_b = uini.reshape(B, -1)
    qp_G = torch.cat([Uf, -Uf, Yf, -Yf], dim=1)
    qp_h = torch.cat([u_upper.repeat_interleave(Uf.shape[1] // u_upper.shape[0]),
                      -u_lower.repeat_interleave(Uf.shape[1] // u_lower.shape[0]),
                      y_upper.repeat_interleave(Yf.shape[1] // y_upper.shape[0]),
                      -y_lower.repeat_interleave(Yf.shape[1] // y_lower.shape[0])]).repeat(B, 1)
    constant = ref.transpose(-2, -1) @ Q_expanded @ ref + lambda_y.unsqueeze(-1).unsqueeze(-1) * yini.transpose(-2, -1) @ yini

    return qp_Q, qp_p, qp_G, qp_h, qp_A, qp_b, constant

## tensor vesion of function oracle_to_qp
def oracle_to_qp_batch(lambda_g: torch.Tensor, lambda_y: torch.Tensor, upred: torch.Tensor, ypred: torch.Tensor,
                Up: torch.Tensor, Uf: torch.Tensor, Yp: torch.Tensor, Yf: torch.Tensor, uini: torch.Tensor, yini: torch.Tensor,
                device: torch.device = torch.device("cuda")):
    """
    Convert batch of Oracle problems to QP problems
    """
    B = uini.shape[0]  # Assuming Uf has shape [B, T, n_u] where B is batch size
    assert lambda_g.ge(0).all() and lambda_y.ge(0).all(), "Regularizers must be non-negative for all batch elements"

    # Handle the case where no regularizer is provided for all batch elements
    if (lambda_g.eq(0) & lambda_y.eq(0)).all():
        print("No regularizer is provided for all batch elements")
        return None
    
    lambda_g = lambda_g.to(device)
    lambda_y = lambda_y.to(device)
    upred = upred.reshape(B, -1, 1).to(device)
    ypred = ypred.reshape(B, -1, 1).to(device)
    Up = Up.unsqueeze(0).repeat(B, 1, 1).to(device)
    Yp = Yp.unsqueeze(0).repeat(B, 1, 1).to(device)
    Uf = Uf.unsqueeze(0).repeat(B, 1, 1).to(device)
    Yf = Yf.unsqueeze(0).repeat(B, 1, 1).to(device)
    uini = uini.reshape(B, -1, 1).to(device)
    yini = yini.reshape(B, -1, 1).to(device)

    qp_Q = 2 * (lambda_g.unsqueeze(-1).unsqueeze(-1) * torch.eye(Uf.shape[-1]).repeat(B, 1, 1).to(device) +
                lambda_y.unsqueeze(-1).unsqueeze(-1) * Yp.transpose(-2, -1) @ Yp)
    qp_p = -2 * (lambda_y.unsqueeze(-1).unsqueeze(-1) * Yp.transpose(-2, -1) @ yini).reshape(B, -1)
    qp_A = torch.cat([Up, Uf, Yf], dim=1)
    qp_b = torch.cat([uini, upred, ypred], dim=1).reshape(B, -1)
    constant = lambda_y.unsqueeze(-1).unsqueeze(-1) * yini.transpose(-2, -1) @ yini

    return qp_Q, qp_p, qp_A, qp_b, constant

def deepc_to_qp_cvxpy(lambda_g: float = 0., lambda_y: float = 0., Q: np.ndarray = None, R: np.ndarray = None, ref: np.ndarray = None,
                Up: np.ndarray = None, Uf: np.ndarray = None, Yp: np.ndarray = None, Yf: np.ndarray = None, uini: cp.Parameter = None, yini: cp.Parameter = None,
                u_upper: np.ndarray = None, u_lower: np.ndarray = None, y_upper: np.ndarray = None, y_lower: np.ndarray = None):
    """
    Convert DeePC problem to QP problem
    """
    assert lambda_g >= 0 and lambda_y >= 0, "Regularizers must be non-negative"
    if math.isclose(lambda_g, 0.) and math.isclose(lambda_y, 0.):
        ## TODO: handle this case
        print("No regularizer is provided")
        return None

    ## expand Q, R to block diagonal matrices
    Q = np.kron(np.eye(Yf.shape[0] // y_upper.shape[0], dtype=float), Q)
    R = np.kron(np.eye(Uf.shape[0] // u_upper.shape[0], dtype=float), R)
    qp_Q = 2 * (Yf.T @ Q @ Yf + Uf.T @ R @ Uf + lambda_y * Yp.T @ Yp + lambda_g * np.eye(Uf.shape[1]))
    qp_p = - 2  * (Yf.T @ Q @ ref + lambda_y * Yp.T @ yini)
    qp_A = Up
    qp_b = uini
    qp_G = np.vstack([Uf, -Uf, Yf, -Yf])
    qp_h = np.concatenate([np.tile(u_upper, (Uf.shape[0] // u_upper.shape[0])), -np.tile(u_lower, (Uf.shape[0] // u_lower.shape[0])), np.tile(y_upper, (Yf.shape[0] // y_upper.shape[0])), -np.tile(y_lower, (Yf.shape[0] // y_lower.shape[0]))])
    constant = ref.T @ Q @ ref + lambda_y * yini.T @ yini
    
    return cp.psd_wrap(qp_Q), qp_p, qp_G, qp_h, qp_A, qp_b, constant