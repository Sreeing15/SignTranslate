import torch
import xgboost as xgb
import numpy as np
import os
from .gnn_model import GNNModel

class HandGesturePredictor:
    def __init__(self, gnn_path=None, xgb_path=None, num_classes=26):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize GNN
        self.gnn = GNNModel().to(self.device)
        self.gnn.eval()
        if gnn_path and os.path.exists(gnn_path):
            self.gnn.load_state_dict(torch.load(gnn_path, map_location=self.device))
            
        # Initialize XGBoost
        self.xgb_clf = xgb.XGBClassifier()
        self.xgb_is_loaded = False
        if xgb_path and os.path.exists(xgb_path):
            self.xgb_clf.load_model(xgb_path)
            self.xgb_is_loaded = True
            
        # Map indices to A-Z
        self.class_map = {i: chr(65 + i) for i in range(26)}
        
    def build_adjacency_matrix(self, edges, num_nodes=21):
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for src, tgt in zip(edges[0], edges[1]):
            adj[src, tgt] = 1.0
            adj[tgt, src] = 1.0
        # Add self connections
        np.fill_diagonal(adj, 1.0)
        return adj
        
    def predict(self, features, edges):
        """
        features: (21, 4) numpy array
        edges: (2, num_edges) numpy array
        """
        if features is None:
            return None
            
        # Prepare inputs
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device) # Batch size 1
        adj = self.build_adjacency_matrix(edges)
        adj = torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get GNN embedding
        with torch.no_grad():
            embedding = self.gnn(x, adj).cpu().numpy()
            
        # Get final classification from XGBoost
        if self.xgb_is_loaded:
            pred_idx = self.xgb_clf.predict(embedding)[0]
            pred_char = self.class_map.get(int(pred_idx), "?")
            
            # Get probability (optional, for thresholding)
            probs = self.xgb_clf.predict_proba(embedding)[0]
            confidence = probs[int(pred_idx)]
            
            if confidence > 0.6: # Threshold
                return pred_char
            return None
            
        # If model isn't trained yet, just return dummy
        # Return none to avoid outputting dummy strings if not trained during general dev
        return None
