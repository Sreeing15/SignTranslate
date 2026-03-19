import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from model.gnn_model import GNNModel
from processing.feature_extractor import FeatureExtractor
from processing.landmark_detector import LandmarkDetector

class ASLDataset(Dataset):
    def __init__(self, data_dir, max_images_per_class=500):
        """
        Expects data_dir to have subfolders A-Z containing images.
        """
        self.data_dir = data_dir
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 1])
        if not self.classes:
            raise ValueError(f"No class folders A-Z found in {data_dir}")
            
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.detector = LandmarkDetector(static_image_mode=True)
        self.extractor = FeatureExtractor()
        
        self.image_paths = []
        self.labels = []
        
        print(f"Scanning dataset directory (max {max_images_per_class} images per class)...")
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            img_paths = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
            
            # Slice the images to avoid long processing times during this run
            for path in img_paths[:max_images_per_class]:
                self.image_paths.append(path)
                self.labels.append(self.class_to_idx[cls_name])
                
        print(f"Found {len(self.image_paths)} images across {len(self.classes)} classes.")

    def process_image(self, img_path):
        frame = cv2.imread(img_path)
        if frame is None: return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, _ = self.detector.detect(frame_rgb)
        
        if not landmarks: return None
        features = self.extractor.extract_features(landmarks)
        return features

    def get_all_valid_features(self):
        """
        Extract features from the whole dataset. 
        Highly recommended to run this once and save to .npy files!
        """
        X = []
        y = []
        print("Extracting MediaPipe landmarks from images... this will take a while!")
        progress = tqdm(total=len(self.image_paths))
        
        for path, label in zip(self.image_paths, self.labels):
            features = self.process_image(path)
            if features is not None:
                X.append(features)
                y.append(label)
            progress.update(1)
            
        progress.close()
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        return torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # On-the-fly gets super slow with mediapipe. 
        # Best to use get_all_valid_features() for GNN
        pass


def train_gnn_real():
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'asl_alphabet_train', 'asl_alphabet_train')
    
    if not os.path.exists(dataset_dir):
        print("ERROR: Could not find Kaggle dataset.")
        print(f"Please extract the ASL Alphabet dataset so the A-Z folders are inside:\n{dataset_dir}")
        return

    dataset = ASLDataset(dataset_dir)
    num_classes = len(dataset.classes)
    
    # 1. Extract Features (Or load if already saved)
    cache_x = 'X_cache.pt'
    cache_y = 'y_cache.pt'
    
    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print("Loading cached features...")
        X_train = torch.load(cache_x)
        y_train = torch.load(cache_y)
    else:
        X_train, y_train = dataset.get_all_valid_features()
        print("Saving features to cache...")
        torch.save(X_train, cache_x)
        torch.save(y_train, cache_y)

    print(f"Total valid hands detected: {len(X_train)}")
    
    # 2. Setup GNN Environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    extractor = FeatureExtractor()
    edges = extractor.get_edge_index()
    adj = np.zeros((21, 21), dtype=np.float32)
    for src, tgt in zip(edges[0], edges[1]):
        adj[src, tgt] = 1.0
        adj[tgt, src] = 1.0
    np.fill_diagonal(adj, 1.0)
    adj_tensor = torch.tensor(adj).unsqueeze(0).to(device)
    
    model = GNNModel().to(device)
    
    class TempClassifier(nn.Module):
        def __init__(self, gnn, num_classes):
            super().__init__()
            self.gnn = gnn
            self.head = nn.Linear(128, num_classes)
            
        def forward(self, x, adj):
            embed = self.gnn(x, adj)
            return self.head(embed)
            
    temp_model = TempClassifier(model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(temp_model.parameters(), lr=0.001)
    
    # 3. Train
    epochs = 20
    batch_size = 64
    print(f"Training GNN for {epochs} epochs on {device}...")
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    temp_model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            batch_adj = adj_tensor.repeat(batch_x.size(0), 1, 1)
            
            outputs = temp_model(batch_x, batch_adj)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = total_loss / total
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {acc:.2f}%")
        
    save_path = os.path.join(os.path.dirname(__file__), 'gnn_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"GNN Feature Extractor saved to {save_path}")

if __name__ == "__main__":
    train_gnn_real()
