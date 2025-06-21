import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# === Generator Model Definition ===
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

# === Load Trained Generator ===
device = torch.device("cpu")
latent_dim = 100
num_classes = 10
generator = Generator(latent_dim, num_classes).to(device)
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()

# === Streamlit UI ===
st.title("🧠 Handwritten Digit Generator")
st.write("Select a digit (0–9) to generate 5 different handwritten images.")

digit = st.selectbox("Choose a digit", list(range(10)))
generate = st.button("Generate Images")

if generate:
    noise = torch.randn(5, latent_dim)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        generated_imgs = generator(noise, labels).numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(generated_imgs[i][0], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
