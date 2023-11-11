import numpy as np
import random

def generate_problem(size: int, max_num: int = 10, deletion_rate: float = 1/3):
    matrix = np.random.randint(1, max_num, size=(size, size))
    solution = np.zeros((size,size))
    for i in range(size):
        solution[i] = (random.choices([0, 1], weights=[deletion_rate, 1 - deletion_rate], k = size))
    row_sum = np.zeros(size)
    col_sum = np.zeros(size)
    for i in range(size):
        row_sum[i] = (np.matmul(matrix[i], np.transpose(solution[i])))
        col_sum[i] = (np.matmul(matrix[:,i], np.transpose(solution[:,i])))
    solution = solution.reshape(1,size ** 2)[0]
    return matrix, row_sum, col_sum

def fitness_binary(candidate: list, matrix, row_sum, col_sum):
    size = matrix.shape[0]
    candidate = np.array(candidate).reshape(size,size)
    print("candidate", candidate)
    for i in range(size):
        if row_sum[i] != np.matmul(matrix[i], np.transpose(candidate[i])):
            print(row_sum[i], np.matmul(matrix[i], np.transpose(candidate[i])))
            #return 0
        if col_sum[i] != np.matmul(matrix[:, i], np.transpose(candidate[:, i])):
            print(col_sum[i], np.matmul(matrix[:, i], np.transpose(candidate[:, i])))
            #return 0
    return 1

size = 3
matrix, row_sum, col_sum = generate_problem(size)
print(matrix, row_sum, col_sum)
candidate = random.choices([0, 1], weights=[0.5, 0.5], k=size**2)
print(fitness_binary(candidate, matrix, row_sum, col_sum))