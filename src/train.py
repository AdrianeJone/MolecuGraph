import torch
from torch_geometric.loader import DataLoader
from src.model import MoleculeGNN
from src.dataset import create_graph

def train_model(smiles_list, target_list, batch_size=64, epochs=10, lr=0.001, hidden_channels=128):
    history = []
    print("Converting SMILES to Graphs...")
    dataset = [create_graph(s, t) for s, t in zip(smiles_list, target_list) if create_graph(s, t) is not None]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Device and Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    model = MoleculeGNN(num_node_features=5, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'model.pth')
    return model, history