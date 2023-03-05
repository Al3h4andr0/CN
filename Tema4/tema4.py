import sys
from collections import defaultdict


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


if __name__ == "__main__":
    # 1
    n, data, rows, cols = read_sparse_matrix("a_1.txt")
    matrix = SparseMatrix(n, data, rows, cols)
    free_terms = read_b("b_1.txt")

    if check_main_diagonal(matrix) is False:
        print("An item on the diagonal is 0")
        sys.exit()


