import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, TensorDataset

def get_olivetti_loader(batch_size=32):
    data = fetch_olivetti_faces(shuffle=True)
    X = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1)  # [n, 1, 64, 64]
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
