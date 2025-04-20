# Import the function that computes similarity between two sequences using various methods
from similarity import compute_sequence_similarity

def compute_hybrid_similarity(ref_seq, live_seq, weights=(0.5, 0.25, 0.25)):
    """
    Compute a hybrid similarity score between a reference sequence and a live sequence using
    a weighted combination of cosine similarity, Euclidean distance, and joint angle similarity.

    Args:
        ref_seq (list of list of floats): The reference pose sequence (frames × flattened keypoints).
        live_seq (list of list of floats): The live pose sequence captured in real time.
        weights (tuple of floats): Weights for combining (cosine, euclidean, angle) similarities.

    Returns:
        float: A final hybrid similarity score between 0 and 1.
    """
    
    # If either sequence is empty, return 0 similarity
    if len(ref_seq) == 0 or len(live_seq) == 0:
        return 0.0

    # Truncate both sequences to the same length (minimum of the two)
    min_len = min(len(ref_seq), len(live_seq))
    ref_seq, live_seq = ref_seq[:min_len], live_seq[:min_len]

    # Compute similarities using three different methods
    cos_sim = compute_sequence_similarity(ref_seq, live_seq, method="cosine")     # similarity in direction
    euc_sim = compute_sequence_similarity(ref_seq, live_seq, method="euclidean")  # similarity in position
    ang_sim = compute_sequence_similarity(ref_seq, live_seq, method="angle")      # similarity in joint angles

    # Unpack the weights and compute the final weighted score
    w_cos, w_euc, w_ang = weights
    final_score = w_cos * cos_sim + w_euc * euc_sim + w_ang * ang_sim

    # Print debug information for monitoring
    print(f"[DEBUG] Cosine: {cos_sim:.3f}, Euclidean: {euc_sim:.3f}, Angle: {ang_sim:.3f}, Hybrid: {final_score:.3f}")

    return final_score
