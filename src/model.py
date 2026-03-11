import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MoleculeGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(MoleculeGNN, self).__init__()
        # Layer 1: GCN Convolution (The "Message Passing" layer)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        # Layer 2
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Layer 3
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Pass through Conv layers with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # 2. Global Pooling: Compress all atoms into one "Molecule Vector"
        x = global_mean_pool(x, batch)
        
        # 3. Final prediction
        x = self.fc(x)
        return x