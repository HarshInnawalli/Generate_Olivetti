import torch
from models.vae import VAE
from utils import plot_samples

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE().to(device)
model.load_state_dict(torch.load("vae_olivetti.pth", map_location=device))
model.eval()

with torch.no_grad():
    z = torch.randn(16, 32).to(device)
    samples = model.decode(z).cpu().numpy()
    plot_samples(samples)
