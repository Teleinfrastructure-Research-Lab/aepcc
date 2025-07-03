import torch
import matplotlib.pyplot as plt

from architecture.gae import InvertedFiLMPositionalEncoding

def visualize_positional_encoding(model, seq_len=50):

    pe_slice = model.pe[0, :, :seq_len]
    pe_slice = pe_slice.T

    plt.figure(figsize=(10, 6))
    plt.imshow(pe_slice, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Encoding Value")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Sequence Position")
    plt.title("Sinusoidal Positional Encoding Heatmap")
    plt.show()

if __name__ == "__main__":
    # Example usage:
    model = InvertedFiLMPositionalEncoding(num_features=128, max_len=2048)
    visualize_positional_encoding(model, seq_len=2048)