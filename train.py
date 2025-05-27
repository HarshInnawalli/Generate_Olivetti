import torch
from torch import optim
from data.loader import get_olivetti_loader
from models.vae import VAE
from utils import vae_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loader = get_olivetti_loader(batch_size=32)
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        x_recon, mu, logvar = model(x_batch)
        loss = vae_loss(x_recon, x_batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader.dataset):.4f}")

# Save model
torch.save(model.state_dict(), "vae_olivetti.pth")
