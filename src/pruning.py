import torch

def dynamic_token_pruning(attention_matrix: torch.Tensor, importance_threshold: float) -> torch.Tensor:
    """
    Applies dynamic token pruning based on token importance.
    In the paper, importance is a cumulative score. Here, we simplify by using
    the max attention score a token gives out as a proxy for its importance.
    Tokens deemed unimportant have all their attention scores (rows) zeroed out.
    """
    # Line 1: Create a copy to avoid modifying the original matrix.
    pruned_matrix = attention_matrix.clone()

    # Line 2: Calculate a proxy for "token importance".
    # We find the single highest attention value in each token's row.
    token_importance = torch.max(attention_matrix, dim=-1).values

    # Line 3: Identify which tokens to prune.
    # This creates a boolean mask: `True` if a token is unimportant, `False` otherwise.
    unimportant_tokens_mask = token_importance < importance_threshold

    # Line 4: Apply the pruning by "masking".
    # This uses the mask to set the entire rows of unimportant tokens to zero.
    pruned_matrix[unimportant_tokens_mask.unsqueeze(-1)] = 0

    return pruned_matrix, unimportant_tokens_mask
