import copy
import sys
from collections import defaultdict

import numpy as np


class SparseMatrix:
    def __init__(self, n, data, rows, cols):
        self.n = n
        self.matrix = defaultdict(lambda: defaultdict(float))
        for i, entry in enumerate(data):
            current_row = rows[i]
            current_col = cols[i]
            self.matrix[current_row][current_col] = entry

    def __getitem__(self, item):
        return self.matrix[item]


def read_sparse_matrix(filename):
    # file of type: n on first line, second line: x, i, j, where x = A[i][j]
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        data, rows, cols = [], [], []
        for line in f:
            parts = [li.strip() for li in line.split(",") if len(li) != 0 and li != '\n']
            if not len(parts):
                continue
            x = float(parts[0].strip())
            i = int(parts[1].strip())
            j = int(parts[2].strip())
            data.append(x)
            rows.append(i)
            cols.append(j)
    return n, data, rows, cols


def check_main_diagonal(matrix: SparseMatrix):
    n = matrix.n
    for i in range(n):
        if matrix[i][i] == 0:
            return False
    return True

def read_b(filename):
    with open(filename, "r") as f:
        n = int(f.readline().strip())

        b = []
        for i in range(n):
            b.append(float(f.readline().strip()))

    return b


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=1000):
    pass
    # # TODO: Change this
    # """
    # Solves the linear system Ax = b using the Gauss-Seidel method.
    #
    # Parameters
    # ----------
    # A : numpy.ndarray
    #     The coefficient matrix of the linear system.
    # b : numpy.ndarray
    #     The right-hand side vector of the linear system.
    # x0 : numpy.ndarray
    #     The initial guess for the solution vector.
    # tol : float, optional
    #     The tolerance for the residual norm. Default is 1e-6.
    # max_iter : int, optional
    #     The maximum number of iterations. Default is 1000.
    #
    # Returns
    # -------
    # numpy.ndarray
    #     The solution vector x.
    #
    # Raises
    # ------
    # ValueError
    #     If the coefficient matrix is not square, or if it is not diagonally dominant.
    #
    # """
    # # Check that the matrix is square
    # n, m = A.shape
    # if n != m:
    #     raise ValueError("Matrix A must be square")
    #
    # # Check that the matrix is diagonally dominant
    # if not np.all(np.abs(A.diagonal()) >= np.sum(np.abs(A), axis=1) - np.abs(A.diagonal())):
    #     raise ValueError("Matrix A must be diagonally dominant")
    #
    # # Initialize the solution vector
    # x = x0.copy()
    #
    # # Perform the iterations
    # for k in range(max_iter):
    #     # Compute the new estimates for each unknown
    #     for i in range(n):
    #         x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    #
    #     # Check the convergence criterion
    #     r = b - np.dot(A, x)
    #     if np.linalg.norm(r) < tol:
    #         break
    #
    # # Return the solution vector
    # return x


if __name__ == "__main__":
    matrixes = []
    bees = []
    # 1
    for i in range(1, 6):
        n, data, rows, cols = read_sparse_matrix("a_{}.txt".format(i))
        matrix = SparseMatrix(n, data, rows, cols)
        free_terms = read_b("b_{}.txt".format(i))
        matrixes.append(matrix)
        bees.append(free_terms)

    for matrix in matrixes:
        if check_main_diagonal(matrix) is False:
            print("An item on the diagonal is 0")
            sys.exit()

    for i, matrix in enumerate(matrixes):
        # x_gs = gauss_seidel(matrix)
        x_gs = copy.deepcopy(bees[i])
        # print(f"Norm for matrix {i}", np.linalg.norm(np.dot(matrix.matrix, x_gs) - bees[i]))


