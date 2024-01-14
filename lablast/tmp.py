import torch

# Original tensor
original_tensor = torch.tensor([[4, 5, 6],
                                [7, 8, 9],
                                [10, 11, 12]])

# Number of times to repeat each row
num_repeats = 3

# Replicate the rows
result_tensor = original_tensor.unsqueeze(-1).expand(-1, -1, 3).transpose(1, 2)

print(result_tensor)
