import numpy as np

# BODY_25 keypoint indices for selected joints
# Format: (start_joint, end_joint)
# These pairs define key limb segments to be embedded
KEYPOINT_PAIRS = [
    (2, 3), (3, 4),  # right upper arm, right lower arm
    (5, 6), (6, 7),  # left upper arm, left lower arm
    (1, 2), (1, 5),  # neck to right shoulder, neck to left shoulder
    (1, 8)           # neck to mid-hip
]

def to_vector(pose):
    """
    Converts a flat list of x, y coordinates to an N x 2 array of 2D points.
    
    Parameters:
        pose (list or np.ndarray): Flat list of keypoints (x0, y0, x1, y1, ..., xn, yn)

    Returns:
        np.ndarray: Array of shape (N, 2)
    """
    return np.array(pose).reshape(-1, 2)

def get_pose_embedding(pose):
    """
    Generates a pose embedding by computing direction vectors between key joint pairs.

    Parameters:
        pose (list or np.ndarray): Flat list of 2D keypoints

    Returns:
        np.ndarray: Flattened vector of selected joint vectors (length = 14 x 2 = 28)
    """
    points = to_vector(pose)

    # Handle the case where all points are zeros (invalid data)
    if np.all(points == 0):
        return np.zeros(50)  # Preserves expected shape for downstream use

    vecs = []
    for a, b in KEYPOINT_PAIRS:
        # If either joint in the pair is missing (i.e., 0), use a zero vector
        if np.any(points[a] == 0) or np.any(points[b] == 0):
            vec = np.zeros(2)
        else:
            vec = points[b] - points[a]  # Compute vector from joint a to b
        vecs.append(vec)
    
    return np.concatenate(vecs)  # Return single flattened embedding vector

def get_joint_angles(pose):
    """
    Computes joint angles for selected segments to measure relative articulation.

    Parameters:
        pose (list or np.ndarray): Flat list of 2D keypoints

    Returns:
        np.ndarray: Vector of 4 angles in radians
    """
    points = to_vector(pose)
    angles = []

    def angle_between(v1, v2):
        """
        Compute the angle between two vectors in radians using the cosine formula.
        """
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        unit1, unit2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
        return np.arccos(dot)

    try:
        # Compute angles at elbows and shoulders
        angles.append(angle_between(points[2] - points[1], points[3] - points[2]))  # R elbow
        angles.append(angle_between(points[3] - points[2], points[4] - points[3]))  # R wrist
        angles.append(angle_between(points[5] - points[1], points[6] - points[5]))  # L elbow
        angles.append(angle_between(points[6] - points[5], points[7] - points[6]))  # L wrist
    except Exception:
        angles = [0.0] * 4  # Fallback in case of missing keypoints

    return np.array(angles)
