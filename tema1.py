import numpy as np

def ex1():
    u = 10
    i = -1
    while True:
        if 1 + u ** i != 1 and 1 + u ** (i - 1) == 1:
            return i
        i = i - 1


def ex2():
    x = 1.0
    y = 10 ** (-15)
    z = 0.1
    print(f'x = {x}, y={y}, z={z}')
    print("(x + y) + z != x + (y + z)?", (x + y) + z != x + (y + z), "\n")
    a = 6
    b = 10 ** -1
    c = 10 ** (-15)
    print(f'a = {a}, b={b}, b={c}')
    # It works because precision error: a * b = 0.6 * c is ok, but b * c is 10e-17 = not ok
    print("(a * b) + c != a * (b * c)?", (a * b) * c != a * (b * c), "\n\n")


def split(arr):
    a = np.zeros((len(arr) // 2, len(arr) // 2))
    b = np.zeros((len(arr) // 2, len(arr) // 2))
    c = np.zeros((len(arr) // 2, len(arr) // 2))
    d = np.zeros((len(arr) // 2, len(arr) // 2))
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i < len(arr) // 2 and j < len(arr) // 2:
                a[i][j] = arr[i][j]
            if i < len(arr) // 2 and j >= len(arr) // 2:
                b[i][j % (len(arr) // 2)] = arr[i][j]
            if i >= len(arr) // 2 and j < len(arr) // 2:
                c[i % (len(arr) // 2)][j] = arr[i][j]
            if i >= len(arr) // 2 and j >= len(arr) // 2:
                d[i % (len(arr) // 2)][j % (len(arr) // 2)] = arr[i][j]
    return a, b, c, d


def concatenate(a, b, c, d, q):
    newMatrix = np.zeros((2 ** (q + 1), 2 ** (q + 1)))
    for i in range(2 ** (q + 1)):
        for j in range(2 ** (q + 1)):
            if i < 2 ** q and j < 2 ** q:
                newMatrix[i][j] = a[i][j]
            if i < 2 ** q and j >= 2 ** q:
                newMatrix[i][j] = b[i][j % (len(newMatrix) // 2)]
            if i >= 2 ** q and j < 2 ** q:
                newMatrix[i][j] = c[i % (len(newMatrix) // 2)][j]
            if i >= 2 ** q and j >= 2 ** q:
                newMatrix[i][j] = d[i % (len(newMatrix) // 2)][j % (len(newMatrix) // 2)]
    return newMatrix


def strassen(A, B, q, qmin):
    if q == qmin:
        return np.matmul(A, B)
    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)
    P1 = strassen(A11 + A22, B11 + B22, q - 1, qmin)
    P2 = strassen(A21 + A22, B11, q - 1, qmin)
    P3 = strassen(A11, B12 - B22, q - 1, qmin)
    P4 = strassen(A22, B21 - B11, q - 1, qmin)
    P5 = strassen(A11 + A12, B22, q - 1, qmin)
    P6 = strassen(A21 - A11, B11 + B12, q - 1, qmin)
    P7 = strassen(A12 - A22, B21 + B22, q - 1, qmin)
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 + P3 - P2 + P6
    return C11, C12, C21, C22


if __name__ == "__main__":
    pwr = ex1()
    print(f"i este : {pwr} \n1+u**i != 1 : {1 + 10 ** pwr != 1} \n1+u**(i-1) != 1 : {1 + 10 ** (pwr - 1) != 1}\n\n")

    ex2()

    A = np.array([[1, 2, 3, 4],
                  [3, 4, 5, 6],
                  [7, 8, 9, 10],
                  [2, 2, 2, 2]])
    B = np.array([[5, 6, 7, 8],
                  [7, 8, 7, 8],
                  [8, 9, 3, 2],
                  [2, 5, 7, 9]])
    C11, C12, C21, C22 = strassen(A, B, 2, 1)
    print(concatenate(C11, C12, C21, C22, 1))
    print(np.matmul(A, B))
