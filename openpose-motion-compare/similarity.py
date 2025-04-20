import numpy as np
from numpy.linalg import norm

# Main function to compute similarity between two pose sequences
def compute_sequence_similarity(ref_seq, live_seq, method="cosine"):
    ref = np.array(ref_seq)
    live = np.array(live_seq)

    # Align the sequence lengths by trimming the longer one
    min_len = min(len(ref), len(live))
    ref = ref[:min_len]
    live = live[:min_len]

    # Cosine similarity: compares the direction of the pose vectors
    if method == "cosine":
        ref_flat = ref.flatten()
        live_flat = live.flatten()
        similarity = np.dot(ref_flat, live_flat) / (norm(ref_flat) * norm(live_flat) + 1e-8)
        return max(0.0, similarity)

    # Euclidean similarity: measures the average normalized spatial difference per frame
    elif method == "euclidean":
        def normalize(seq):
            seq = np.reshape(seq, (len(seq), -1, 2))  # Reshape to (frames, joints, 2)
            seq -= np.mean(seq, axis=1, keepdims=True)  # Center joint positions per frame
            seq /= (np.std(seq, axis=1, keepdims=True) + 1e-8)  # Normalize scale
            return seq.reshape(len(seq), -1)  # Flatten back

        ref_norm = normalize(ref)
        live_norm = normalize(live)

        # Compute per-frame Euclidean distances and average them
        distances = [norm(a - b) for a, b in zip(ref_norm, live_norm)]
        avg_dist = np.mean(distances)
        similarity = max(0.0, 1 - avg_dist / 10.0)  # Scale factor to convert distance to similarity
        return similarity

    # Angle similarity: compares joint angles (e.g., shoulder, hip) across sequences
    elif method == "angle":
        def extract_vectors(pose):
            pose = np.reshape(pose, (-1, 2))  # Reshape to (joints, 2)
            # Define joint triplets (a, b, c) for angle at b between vectors ba and bc
            pairs = [(2, 1, 8),  # right shoulder
                     (5, 4, 8),  # left shoulder
                     (9, 8, 1),  # right hip
                     (12, 11, 1)]  # left hip
            angles = []
            for a, b, c in pairs:
                if max(a, b, c) >= len(pose):
                    continue
                ba = pose[a] - pose[b]
                bc = pose[c] - pose[b]
                cosine_angle = np.dot(ba, bc) / (norm(ba) * norm(bc) + 1e-8)
                angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure it's a valid cosine value
                angles.append(angle)
            return np.array(angles)

        # Compute angles for each frame and get similarity from cosine between angle vectors
        ref_angles = np.array([extract_vectors(frame) for frame in ref])
        live_angles = np.array([extract_vectors(frame) for frame in live])
        angle_sim = np.mean(np.dot(ref_angles, live_angles.T) /
                            (norm(ref_angles, axis=1, keepdims=True) *
                             norm(live_angles, axis=1, keepdims=True).T + 1e-8))
        return max(0.0, angle_sim)

    else:
        raise ValueError("Invalid similarity method")
