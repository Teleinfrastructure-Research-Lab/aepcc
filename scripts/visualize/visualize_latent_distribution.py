import torch
import click
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal, kl_divergence

from smol.core import smol
from smol.register.Architecture import CommonArchitecture
from smol.register.DataLoader import CommonDataLoader

def accumulate_latents(dataloader, model, device):
    accumulated_z = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            batch.to(device)
            out = model(batch)
            z = out["z"].detach().cpu()
            accumulated_z.append(z)

    accumulated_z = torch.cat(accumulated_z, dim=0)
    return accumulated_z

def analyze_latents(latents, output_dir):
    latents_np = latents.numpy()
    means = latents_np.mean(axis=0)
    stds = latents_np.std(axis=0)
    mins = latents_np.min(axis=0)
    maxs = latents_np.max(axis=0)

    stats_array = np.stack([means, stds, mins, maxs], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "latent_stats.npy")
    np.save(stats_path, stats_array)

    print(f"Saved stats array to {stats_path}")

    plt.figure(figsize=(8,6))
    plt.hist(latents_np.flatten(), bins=100, density=True)
    plt.title("Latent Space Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(os.path.join(output_dir, "latent_distribution_histogram.png"))

    return means, stds

def compute_kl_divergence(means, stds, output_dir):
    n_dims = len(means)
    kl_matrix = torch.zeros((n_dims, n_dims))

    for i in range(n_dims):
        for j in range(n_dims):
            if i != j:
                p = Normal(means[i], stds[i] + 1e-6)
                q = Normal(means[j], stds[j] + 1e-6)
                kl = kl_divergence(p, q)
                kl_matrix[i, j] = kl

    kl_df = pd.DataFrame(kl_matrix.numpy())
    kl_path = os.path.join(output_dir, "latent_kl_divergence.csv")
    kl_df.to_csv(kl_path, index_label="dimension_i", header=[f"dimension_{j}" for j in range(n_dims)])

    print(f"Saved KL divergence matrix to {kl_path}")

@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def calculate_latent_distribution(input_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smol.register()
    smol.add_runtime_configs([os.path.normpath(input_path)])
    exp_name = os.path.basename(input_path).split('.')[:-1][0]

    arch_name = smol.get_config(exp_name, "architecture", "name")
    arch_params = smol.get_config(exp_name, "architecture", "module_params")
    arch_load_ckpt = smol.get_config(exp_name, "architecture", "load_ckpt")
    arch = CommonArchitecture(arch_name, arch_params)
    arch.to(device)
    arch.load_checkpoints(*arch_load_ckpt.values())
    arch.eval()

    dl_name = smol.get_config(exp_name, "dataloader-train", "name")
    dl_params = smol.get_config(exp_name, "dataloader-train", "dl_params")
    dl = CommonDataLoader(dl_name, dl_params)

    latents = accumulate_latents(dl, arch, device)

    output_dir = os.path.join("outputs/latent_analysis", exp_name)
    means, stds = analyze_latents(latents, output_dir)
    compute_kl_divergence(means, stds, output_dir)

if __name__ == "__main__":
    calculate_latent_distribution()
