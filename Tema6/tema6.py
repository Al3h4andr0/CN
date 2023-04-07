import numpy as np
from sys import *

# Function to solve the linear system for the least squares method
def solve_this(A, b):
    return np.linalg.solve(A, b)


def generate_nodes(n, x0, xn):
    xi = np.sort(np.random.uniform(x0, xn, n - 1))
    return np.concatenate(([x0], xi, [xn]))


def generate_function_values(nodes, func):
    return func(nodes)


def lagrange_interpolation(nodes, function_values, x):
    n = len(nodes) - 1
    F = np.zeros((n + 1, n + 1))
    F[:, 0] = function_values

    for j in range(1, n + 1):
        for i in range(j, n + 1):
            F[i, j] = ((x - nodes[i - j]) * F[i, j - 1] - (x - nodes[i]) * F[i - 1, j - 1]) / (nodes[i] - nodes[i - j])

    return F[n, n]


# Compute the polynomial approximation using the method of least squares
def least_squares(nodes, function_values, m):
    n = len(nodes)
    A = np.zeros((n, m + 1))

    for i in range(n):
        for j in range(m + 1):
            A[i, j] = nodes[i] ** j

    b = function_values.reshape((n, 1))

    c = solve_this(A.T @ A, A.T @ b)

    return c.flatten()


def horner_evaluation(coefficients, x):
    y = coefficients[-1]

    for i in range(len(coefficients) - 2, -1, -1):
        y = coefficients[i] + x * y

    return y


# Define the function to be interpolated
def f(x):
    return np.sin(x)


# Define the interpolation bounds and the point at which to evaluate the polynomial
with open('numbers.txt', 'r') as file:
    line = file.readline()
    x0, xn = map(float, line.split())

x_complement = np.pi / 4

# Generate the interpolation nodes and the function values
n = 10
nodes = generate_nodes(n, x0, xn)
function_values = generate_function_values(nodes, f)

# Compute the Lagrange interpolation polynomial and its absolute error
L_n = lagrange_interpolation(nodes, function_values, x_complement)
L_n_error = abs(L_n - f(x_complement))
print("Lagrange interpolation:")
print(f"L_{n}(x_complement) = {L_n}")
print(f"|L_{n}(x_complement) - f(x_complement)| = {L_n_error}")

# Compute the polynomial approximation using the method of least squares
m_values = [2, 3, 4, 5]
for m in m_values:
    coefficients = least_squares(nodes, function_values, m)
    P_m = horner_evaluation(coefficients, x_complement)
    P_m_error = abs(P_m - f(x_complement))
    sum_squared_errors = 0
    for i in range(n):
        sum_squared_errors += (horner_evaluation(coefficients, nodes[i]) - function_values[i]) ** 2

    # Print the results
    print(f"\nPolynomial approximation (m={m}):")
    print(f"P_{m}(x_complement) = {P_m}")
    print(f"|P_{m}(x_complement) - f(x_complement)| = {P_m_error}")
    print(f"Sum of squared errors = {sum_squared_errors}")
