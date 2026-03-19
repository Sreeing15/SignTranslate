import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj):
        # h: (batch, N, in_features)
        # adj: (batch, N, N)
        batch_size = h.size(0)
        N = h.size(1)

        Wh = self.W(h) # (batch, N, out_features)
        
        # Prepare for attention calculation
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1) # (batch, N, N, out_features)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1) # (batch, N, N, out_features)
        
        # Concatenate and pass through attention mechanism
        a_input = torch.cat([Wh1, Wh2], dim=-1) # (batch, N, N, 2 * out_features)
        e = self.leakyrelu(self.a(a_input).squeeze(-1)) # (batch, N, N)
        
        # Masked attention: only consider connected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        
        # Aggregate features
        h_prime = torch.bmm(attention, Wh) # (batch, N, out_features)
        return F.elu(h_prime)

class GNNModel(nn.Module):
    def __init__(self, num_nodes=21, in_features=4, hidden_dim=64, embed_dim=128):
        super(GNNModel, self).__init__()
        self.gat1 = GraphAttentionLayer(in_features, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x, adj):
        # x: (batch, N, in_features)
        # adj: (batch, N, N)
        
        h = self.gat1(x, adj)
        h = self.gat2(h, adj)
        
        # Flatten graph representation
        h = h.view(h.size(0), -1)
        
        embedding = self.fc(h)
        return embedding
