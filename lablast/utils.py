import torch


def shift_2d_to_3d(original_matrix, n):
    repeated_matrix = original_matrix.unsqueeze(-1).expand(-1, -1, n)
    rows, cols = original_matrix.size()
    shift_indices = (torch.arange(n).unsqueeze(1) + torch.arange(cols)) % cols
    #repeat the matrix along first dim
    # shift_indices = shift_indices.unsqueeze(0).expand(repeated_matrix.shape[0], -1, -1)
    shifted_matrices = original_matrix[:, shift_indices].transpose(1, 2)
    return shifted_matrices



def min_2nd_3rd(matrix_3d):
    reshaped_matrix = matrix_3d.view(matrix_3d.shape[0], -1)

    min_indices = torch.argmin(reshaped_matrix, dim=1)
    i_indices = min_indices // matrix_3d.shape[2]
    j_indices = min_indices % matrix_3d.shape[2]

    return i_indices, j_indices
