import torch

torch.manual_seed(123213)

if __name__ == "__main__":
    #distances random 2d 10x10
    d_positions = torch.rand(10, 2)
    distances = torch.cdist(d_positions, d_positions, p=2)
    print(distances)
    # Original matrix
    original_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Number of times to repeat the matrix
    n = 5

    # Repeat the matrix along a new dimension
    repeated_matrix = original_matrix.repeat(n, 1, 1)

    # Generate indices for shifting
    rows, cols = original_matrix.size()
    shift_indices = (torch.arange(n).unsqueeze(1) + torch.arange(cols)) % cols

    # Use advanced indexing to create the shifted matrices
    shifted_matrices = repeated_matrix[torch.arange(n).unsqueeze(-1), :, shift_indices].transpose(1, 2)
    dis = distances[original_matrix, shifted_matrices]
    for i in range(len(dis)):
        print(dis[i], original_matrix, shifted_matrices[i])
    # for matrix in shifted_matrices:
    #     print(matrix)
