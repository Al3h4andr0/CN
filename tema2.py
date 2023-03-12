import numpy as np
import copy
import scipy.linalg as sc


# def determinant(A):
#     n = A.shape[0]
#     if n == 1:
#         return A[0, 0]
#     det = 0
#     for j in range(n):
#         A_sub = np.delete(np.delete(A, 0, axis=0), j, axis=1)
#         det += ((-1) ** j) * A[0, j] * determinant(A_sub)
#     return det

def minor(A, i, j):
    submatrix = copy.deepcopy(A)
    submatrix = [list(row) for it, row in enumerate(submatrix) if it != i]
    submatrix = [list(col) for it, col in enumerate(zip(*submatrix)) if it != j]
    return np.asarray(submatrix)

def is_positive_definite(A):
    n, m = A.shape
    for i in reversed(range(n)):
        submatrix = minor(A, i, i)
        if np.linalg.det(submatrix) < 0:
            return False
    return True

def ldl_decomposition(A):
    n = A.shape[0]
    L = np.identity(n)
    D = np.zeros(n)

    for j in range(n):
        for i in range(j, n):
            s = 0
            for k in range(j):
                s += L[i, k] * D[k] * L[j, k]
            if i == j:
                D[j] = A[j, j] - s
            else:
                L[i, j] = (A[i, j] - s) / D[j]

    return L, D


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    return y


def backward_substitution(L_transpose, z):
    n = L_transpose.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += L_transpose[i, j] * x[j]
        x[i] = (z[i] - s) / L_transpose[i, i]
    return x


def ldl_solver(A, b):
    Ainit = copy.deepcopy(A)
    L, D = ldl_decomposition(Ainit)

    y = forward_substitution(L, b)
    z = y / D
    L_transpose = L.T
    x = backward_substitution(L_transpose, z)
    return x


if __name__ == "__main__":
    print("INCEPEM")
    # A = np.array([[1, -1, 2], [-1, 5, -4], [2, -4, 6]])
    # b = np.array([2, 5, 6])
    n = 100
    A = np.random.uniform(-10, 10, size=(n, n))
    A = 0.5 * (A + A.T)
    b = np.random.uniform(-10, 10, size=n)
    is_positive = is_positive_definite(A)

    if not is_positive or len(A) != len(A[0]):
        print("Matrix is not positive definite or is not quadratic")
    else:
        print("OK")

    while not is_positive:
        print("Looking for another matrix...")
        A = np.random.uniform(-10, 10, size=(n, n))
        A = 0.5 * (A + A.T)
        b = np.random.uniform(-10, 10, size=n)
        is_positive = is_positive_definite(A)
    print("Found a matrix.")

    L, D = ldl_decomposition(copy.deepcopy(A))

    print("L:", L, "\nD:", D, "\n")
    print("det(A) = ", np.linalg.det(L) * np.linalg.det(np.diag(D)) * np.linalg.det(L.T))
    our_x = ldl_solver(copy.deepcopy(A), b)
    print("RESULT: ", our_x)

    x = sc.solve(copy.deepcopy(A), b)
    print("Expected result:", x)
    print("Norm (Verification): ", np.linalg.norm((np.dot(A, our_x) - b)))
