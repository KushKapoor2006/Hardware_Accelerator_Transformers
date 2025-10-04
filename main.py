# Import the functions you've already built
from src.attention_model import create_attention_matrix, calculate_sparsity
from src.pruning import dynamic_token_pruning

def main():
    """
    Main function to run the SpAtten proof-of-concept model.
    This script simulates a dense attention matrix, applies our pruning
    algorithm, and analyzes the quantifiable impact on computation and memory access.
    """
    # --- 1. Define Assumed & Consistent Parameters ---
    # These parameters define the scale of our simulation. They are chosen
    # to be representative of a common Transformer model like BERT-Base.
    BATCH_SIZE = 1        # How many sentences to process at once.
    SEQ_LENGTH = 256      # The number of tokens (words) in each sentence.
    NUM_HEADS = 12        # The number of parallel attention heads.
    IMPORTANCE_THRESHOLD = 0.05  # Our key hyperparameter for pruning.

    print("--- SpAtten: Proof-of-Concept for Dynamic Token Pruning ---")
    print(f"Parameters: Batch Size={BATCH_SIZE}, Sequence Length={SEQ_LENGTH}, Heads={NUM_HEADS}\n")

    # --- 2. Generate the Baseline Dense Attention Matrix ---
    print("Step 1: Simulating a dense attention matrix from a Transformer model...")
    dense_matrix = create_attention_matrix(SEQ_LENGTH, NUM_HEADS, BATCH_SIZE)

    # --- 3. Apply the Pruning Algorithm ---
    print(f"\nStep 2: Applying dynamic token pruning with importance threshold = {IMPORTANCE_THRESHOLD}...")
    pruned_matrix, mask = dynamic_token_pruning(dense_matrix, IMPORTANCE_THRESHOLD)
    
    # --- 4. Calculate & Report Quantifiable Metrics ---
    final_sparsity = calculate_sparsity(pruned_matrix)
    num_pruned_tokens = mask.sum().item()
    total_tokens_in_heads = mask.numel()
    pruned_token_percentage = (num_pruned_tokens / total_tokens_in_heads) * 100

    print(f" -> Tokens pruned: {num_pruned_tokens} out of {total_tokens_in_heads} total tokens across all heads ({pruned_token_percentage:.2f}%)")
    print(f" -> Final Matrix Sparsity: {final_sparsity:.2f}%")

    # --- 5. Analyze the Impact on Hardware Performance ---
    print("\n--- Hardware Impact Analysis ---")
    
    # In one head, the number of Multiply-Accumulate (MAC) operations for the
    # attention_scores * Value multiplication is roughly SEQ_LENGTH * SEQ_LENGTH.
    total_ops_per_head = SEQ_LENGTH * SEQ_LENGTH
    
    # After pruning, we only perform operations for the rows that were NOT pruned.
    num_active_tokens_per_head = (total_tokens_in_heads - num_pruned_tokens) / NUM_HEADS
    pruned_ops_per_head = int(num_active_tokens_per_head * SEQ_LENGTH)
    
    ops_reduction_percentage = ((total_ops_per_head - pruned_ops_per_head) / total_ops_per_head) * 100

    print(f"For a single attention head:")
    print(f" -> Baseline (Dense) Operations: ~{total_ops_per_head:,} MACs")
    print(f" -> Pruned (Sparse) Operations: ~{pruned_ops_per_head:,} MACs")
    print(f" -> Reduction in Compute Operations: {ops_reduction_percentage:.2f}%")

    # This is the most crucial connection, linking the algorithm to hardware efficiency.
    memory_access_reduction_factor = total_ops_per_head / pruned_ops_per_head if pruned_ops_per_head > 0 else float('inf')
    
    print(f"\nThis compute reduction directly correlates to a **~{memory_access_reduction_factor:.1f}x reduction in off-chip DRAM access**.")
    print("This is because the pruned rows of the Key and Value matrices no longer need to be fetched,")
    print("demonstrating the core principle of the SpAtten hardware-software co-design.")


if __name__ == "__main__":
    main()
