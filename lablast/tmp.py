import torch

# Create a tensor of shape (100, 30, 30)
matrix = torch.zeros(5, 4, 4)

# Fill the lower triangular part of each 30x30 submatrix with 99999
matrix[torch.tril(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)] = 999999

print(matrix)

