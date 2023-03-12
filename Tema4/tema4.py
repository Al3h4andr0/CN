import copy
import sys
from collections import defaultdict

import numpy
import numpy as np

class SparseMatrix:
    def __init__(self, n, data, rows, cols):
        self.n = n
        self.matrix = defaultdict(lambda: defaultdict(numpy.float64))
        for i, entry in enumerate(data):
            current_row = rows[i]
            current_col = cols[i]
            self.matrix[current_row][current_col] = entry

    def __getitem__(self, item):
        return self.matrix[item]

    def __len__(self):
        return self.n

# class SparseMatrix:
#
#     # class RowWrapper:
#     #     def __init__(self, arr, columns, row):
#     #         self.arr = arr
#     #         self.columns = columns
#     #         self.row = row
#     #
#     #     def __getitem__(self, row_item):
#     #         if row_item in self.columns:
#     #             return self.arr[self.columns.index(row_item)]
#     #         else:
#     #             return 0
#
#     def __init__(self, n, data, rows, cols):
#         self.n = n
#         self.data = data
#         self.rows = rows
#         self.cols = cols
#
#     def __getitem__(self, item):
#         line_positions = [i for i, element in enumerate(self.rows) if element == item]
#         column_positions = [self.cols[i] for i in line_positions]
#         row_data = [self.data[i] for i in line_positions]
#         wrapper = self.RowWrapper(row_data, column_positions, item)
#         return wrapper
#
#     def __len__(self):
#         return self.n


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
        try:
            if matrix[i][i] == 0:
                return False
        except Exception as e:
            print(e)
    return True

def read_b(filename):
    with open(filename, "r") as f:
        n = int(f.readline().strip())

        b = []
        for i in range(n):
            b.append(float(f.readline().strip()))

    return b


def gauss_seidel(A, b, tol=1e-10, max_iter=100):
    # Check the dimensions of A and b
    n = len(A)
    x0 = np.zeros(n, dtype=np.float64)
    if len(b) != n:
        raise ValueError("Incompatible dimensions")

    # Initialize the solution vector x
    x = np.copy(x0)

    # Perform the Gauss-Seidel iteration
    for k in range(max_iter):
        # Update each component of x in turn
        for i in range(n):
            # Compute the new value of x[i] using the current values of x and the elements of A and b
            Ax = sum(A[i][j] * x[j] for j in A[i] if j != i)
            x[i] = (b[i] - Ax) / A[i][i]

        # Compute the relative error in the solution and check if it is below the tolerance
        r = np.linalg.norm(mat_vec_mult(A, x) - b) / np.linalg.norm(b)
        if r < tol:
            break

    # Check if the maximum number of iterations was reached
    if k == max_iter - 1:
        print("Warning: maximum number of iterations reached")

    # Return the solution vector x
    return x


def mat_vec_mult(A, x):
    # Check the dimensions of A and x
    n = len(A)
    if x.size != n:
        raise ValueError("Incompatible dimensions")

    # Compute the product of A and x
    Ax = np.zeros(n, dtype=numpy.float64)
    for i in range(n):
        for j in A[i]:
            Ax[i] += A[i][j] * x[j]

    # Return the result
    return Ax


if __name__ == "__main__":
    matrixes = []
    bees = []
    # 1
    for i in range(1, 6):
        n, data, rows, cols = read_sparse_matrix("a_{}.txt".format(i))
        matrix = SparseMatrix(n, data, rows, cols)
        free_terms = read_b("b_{}.txt".format(i))
        print("a_{}.txt = ".format(i), len(matrix))
        print("b_{}.txt".format(i), f'={len(free_terms)}')
        matrixes.append(matrix)
        bees.append(free_terms)

    for matrix in matrixes:
        if check_main_diagonal(matrix) is False:
            print("An item on the diagonal is 0")
            sys.exit()
    for i, matrix in enumerate(matrixes):
        x_gs = gauss_seidel(matrix, bees[i])
        print(f"Norm for matrix {i+1}", np.linalg.norm(mat_vec_mult(matrix, x_gs) - bees[i]))


