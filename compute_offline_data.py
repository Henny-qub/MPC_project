import numpy as np
import scipy.linalg as spla
from DP.offline_matrices import *

N = 5
A = np.array([[1.1, 0.9],
              [0.5, 1.5]])
B = np.array([[2], [0]])
f = np.array([[0], [0]])

Q = np.eye(2)
R = np.array([[1]])
F_x = np.array([[1, 1]])
F_u = np.array([[1]])
FN = 6 * np.eye(2)

compute_offline_matrices(Q, R, F_x, F_u, FN, A, B, f,N)
# compute_value_functions(Q, R, F_x, F_u, B, A, f, y, H)
