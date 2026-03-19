import numpy as np

class FeatureExtractor:
    def __init__(self):
        # We define the graph structure for the GNN here
        # Edges based on hand anatomy (MediaPipe connections)
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
            (5, 9), (9, 13), (13, 17)                # Palm base connections
        ]

    def extract_features(self, landmarks):
        """
        Convert 21x3 landmarks list to structured features.
        Returns node features (N_nodes x feature_dim) and edge attributes if needed.
        """
        if not landmarks or len(landmarks) != 21:
            return None
        
        landmarks = np.array(landmarks)
        
        # Normalize landmarks relative to wrist (node 0)
        wrist = landmarks[0]
        normalized_landmarks = landmarks - wrist
        
        # Make the features scale-invariant by dividing by the maximum distance to wrist
        max_dist = np.max(np.linalg.norm(normalized_landmarks, axis=1))
        if max_dist > 0:
            normalized_landmarks = normalized_landmarks / max_dist
            
        # Calculate distance to wrist as an additional feature
        distances = np.linalg.norm(normalized_landmarks, axis=1, keepdims=True)
        
        # Node features: Shape (21, 4) -> [x, y, z, dist_to_wrist]
        node_features = np.hstack((normalized_landmarks, distances))
        
        return node_features
        
    def get_edge_index(self):
        """
        Returns edge index in shape [2, num_edges] for PyTorch Geometric
        or manual message passing. Includes bidirectional edges.
        """
        source = [e[0] for e in self.edges] + [e[1] for e in self.edges]
        target = [e[1] for e in self.edges] + [e[0] for e in self.edges]
        return np.array([source, target], dtype=np.int64)
