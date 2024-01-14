import torch
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    A = torch.rand(1000, 100).to(device)

    # Indices i and j for each row
    i_indices = torch.randint(0, A.size(1), (A.size(0),), device=device)
    j_indices = torch.randint(0, A.size(1), (A.size(0),), device=device)
    # j indices should be bigger
    tmp = j_indices
    j_indices = torch.max(i_indices, j_indices)
    i_indices = torch.min(tmp, i_indices)

    for i in tqdm(range(1000000)):
        # Calculate lengths of segments to be reversed
        lengths = j_indices - i_indices

        # Creating a range tensor
        range_tensor = torch.arange(A.size(1), device=device).unsqueeze(0).expand_as(A)

        # Calculate corrected reversed indices
        corrected_indices = i_indices.unsqueeze(1) + lengths.unsqueeze(1) - 1 - (range_tensor - i_indices.unsqueeze(1))
        corrected_indices = torch.clamp(corrected_indices, 0, A.size(1) - 1)  # Clamping to avoid out-of-bounds

        # Creating mask for elements to be reversed
        mask = (range_tensor >= i_indices.unsqueeze(1)) & (range_tensor < j_indices.unsqueeze(1))

        # Reversing the elements
        A[mask] = torch.gather(A, 1, corrected_indices)[mask]
