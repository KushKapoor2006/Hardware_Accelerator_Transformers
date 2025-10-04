import torch
import torch.nn.functional as F

# --- Function 1: create_attention_matrix ---
def create_attention_matrix(seq_len: int, num_heads: int, batch_size: int) -> torch.Tensor:
    """
    Simulates the output of a multi-head attention mechanism.
    This creates a semi-realistic attention matrix by generating random data
    and applying softmax to simulate the probability distribution.
    """
    # Line 1: Simulate raw scores from the QK^T operation.
    # The result is a 4D tensor: (batch_size, num_heads, seq_len, seq_len)
    raw_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

    # Line 2: Apply the softmax function.
    # This converts the raw scores into probabilities, ensuring each row sums to 1.
    attention_matrix = F.softmax(raw_scores, dim=-1)

    return attention_matrix

# --- Function 2: calculate_sparsity ---
def calculate_sparsity(matrix: torch.Tensor) -> float:
    """
    Calculates the percentage of elements in a tensor that are exactly zero.
    """
    # Line 3: Count the number of elements that are exactly 0.
    num_zeros = (matrix == 0).sum().item()
    # Line 4: Get the total number of elements in the tensor.
    total_elements = matrix.numel()
    # Line 5: Calculate and return the sparsity as a percentage.
    sparsity = (num_zeros / total_elements) * 100
    return sparsity
