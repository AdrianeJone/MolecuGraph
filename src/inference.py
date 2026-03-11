import torch
from src.model import MoleculeGNN
from src.dataset import create_graph

num_node_features = 5 # 4 for atom types (H, C, N, O) + 1 for degree
hidden_channels = 128

def predict_property(smiles):
    model = MoleculeGNN(num_node_features=num_node_features, hidden_channels=hidden_channels)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()
    
    with torch.no_grad():
        graph = create_graph(smiles)
        if graph is None:
            raise ValueError("Failed to create graph for the given SMILES string.")
        graph = graph.to(next(model.parameters()).device)
        prediction = model(graph)
    
    return prediction.item()