import numpy as np
import scipy.linalg as spla

def compute_offline_matrices(Q, R, F_x, F_u, FN, A, B, f,N):
    """
    Computes the value functions P_t and p_t for a finite-horizon LQR problem.
        Q (numpy.ndarray): The state cost matrix of shape (n, n).
        R (numpy.ndarray): The control cost matrix of shape (m, m).
        F_x (numpy.ndarray): The state dynamics matrix of shape (n, n).
        F_u (numpy.ndarray): The control dynamics matrix of shape (n, m).
        B (numpy.ndarray): The control input matrix of shape (n, 1).
        A (numpy.ndarray): The state transition matrix of shape (n, n).
        f (numpy.ndarray): The state offset vector of shape (n, 1).

    Returns:
        P (List[numpy.ndarray]): A list of P_t matrices of shape (n, n) for t=0,...,N.
        p (List[numpy.ndarray]): A list of p_t vectors of shape (n, 1) for t=0,...,N.
    """
    # Compute H
    Fxu = np.hstack((F_x, F_u))
    FxuKron = np.kron(np.eye(N - 1), Fxu)
    H = spla.block_diag(FxuKron, FN)

    # Get the problem dimensions
    n = Q.shape[0]
    m = R.shape[0]

    # Define the value functions
    P = [None] * (N + 1)
    K = [None] * N
    p = [None] * (N + 1)

    # Terminal value function
    P[N] = Q
    p[N] = np.zeros((n, 1))

    # Recursive computation of value functions
    for t in range(N - 1, -1, -1):
        print(t)
        P[t] = Q + F_x.T @ P[t + 1] @ A @ F_x
        # Rtilde = R + B.T @ P[t] @ B
        # p[t] = F_x.T @ p[t + 1] + F_x.T @ P[t + 1] @ (B @ R @ B.T + F_u.T @ P[t + 1] @ F_u) @ f
        # p[t] += y[t].T @ H @ f

    return P, p
