import os
import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from model.gnn_model import GNNModel
from processing.feature_extractor import FeatureExtractor

def train_xgboost_real():
    print("Initialize XGBoost training pipelines on REAL dataset...")
    
    # Check for cache
    cache_x = 'X_cache.pt'
    cache_y = 'y_cache.pt'
    
    if not (os.path.exists(cache_x) and os.path.exists(cache_y)):
        print("Dataset Cache not found. Please run train_gnn.py first so it can process the MediaPipe landmarks.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gnn_path = os.path.join(os.path.dirname(__file__), 'gnn_model.pth')
    if not os.path.exists(gnn_path):
        print("GNN model weights not found. Run train_gnn.py first.")
        return
        
    # 1. Setup GNN
    model = GNNModel().to(device)
    model.load_state_dict(torch.load(gnn_path, map_location=device))
    model.eval()
    
    # 2. Load Features
    X_all = torch.load(cache_x).to(device)
    y_all = torch.load(cache_y).numpy()
    
    extractor = FeatureExtractor()
    edges = extractor.get_edge_index()
    adj = np.zeros((21, 21), dtype=np.float32)
    for src, tgt in zip(edges[0], edges[1]):
        adj[src, tgt] = 1.0
        adj[tgt, src] = 1.0
    np.fill_diagonal(adj, 1.0)
    adj_tensor = torch.tensor(adj).unsqueeze(0).to(device)
    
    # 3. Extract Embeddings in batches to avoid OOM
    print("Extracting dense representations using trained GNN...")
    embeddings = []
    
    batch_size = 128
    with torch.no_grad():
        for i in range(0, X_all.size(0), batch_size):
            batch_x = X_all[i:i+batch_size]
            batch_adj = adj_tensor.repeat(batch_x.size(0), 1, 1)
            emb = model(batch_x, batch_adj).cpu().numpy()
            embeddings.extend(emb)
            
    embeddings = np.array(embeddings)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_all, test_size=0.2, random_state=42)
    
    # 5. Train XGBoost
    print(f"Training XGBoost Classifier on {len(X_train)} samples...")
    num_classes = len(np.unique(y_all))
    
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        max_depth=6,
        n_estimators=150,
        learning_rate=0.1,
        tree_method='hist', # Much faster for large datasets
    )
    
    clf.fit(X_train, y_train)
    
    # 6. Evaluation
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Real Test Accuracy: {acc * 100:.2f}%")
    
    xgb_save_path = os.path.join(os.path.dirname(__file__), 'xgb_model.json')
    clf.save_model(xgb_save_path)
    print(f"XGBoost model saved to {xgb_save_path}")

if __name__ == "__main__":
    train_xgboost_real()
