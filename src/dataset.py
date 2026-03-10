import torch
from torch_geometric.data import Data
from rdkit import Chem

def create_graph(smiles, target_value):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
    
    features = []
    allowed_atomic_nums = [1, 6, 7, 8]  # H, C, N, O
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        feat = [0] * len(allowed_atomic_nums)  # Initialize feature vector
        if atomic_num in allowed_atomic_nums:
            feat[allowed_atomic_nums.index(atomic_num)] = 1  # One-hot encoding for the atom type
        feat.append(atom.GetDegree())
        features.append(feat)
        
    x = torch.tensor(features, dtype=torch.float)
    
    rows, cols = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows.extend([i, j])
        cols.extend([j, i])
    
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=torch.tensor([target_value], dtype=torch.float))