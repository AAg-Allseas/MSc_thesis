
"""Visualization utilities for MIONet model outputs."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset
from src.prototyping.deepOnet.model_deepOnet import MIONet
from src.prototyping.deepOnet.models import model_2, model_cnn_1, model_cnn_2
from src.prototyping.deepOnet.utils import BranchConstructor, MLPConstructor, prepare_batch



def plot_prediction(
    checkpoint_path: str | Path,
    model: MIONet,
    data_path: Optional[str | Path] = None,
    sample_idx: int = 0,
    device: Optional[torch.device] = None,
) -> plt.Figure:
    """Plot model prediction vs ground truth for all features.

    Args:
        checkpoint_path: Path to the saved checkpoint (.pth file).
        data_path: Path to data directory. Defaults to standard location.
        sample_idx: Index of sample in the dataset to plot.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        Matplotlib figure with prediction and ground truth for all features.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if data_path is None:
        data_path = Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data")
    else:
        data_path = Path(data_path)

    # Load model
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data
    files = find_parquet_files(
        data_path,
        lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] > 40
    )
    
    sample_length = 10000
    feats_sensors = ['tau_ext_x', 'tau_ext_y', 'tau_ext_mz']
    scales_sensors = np.array([1/75e3, 1/75e3, 1/100e3])
    feats_samples = [
        'pos_eta_x', 'pos_eta_y', 'pos_eta_mz',
        'pos_nu_x', 'pos_nu_y', 'pos_nu_mz',
        'rpm_bow_fore', 'rpm_bow_aft', 'rpm_stern_fore', 'rpm_stern_aft',
        'rpm_fixed_ps', 'rpm_fixed_sb',
    ]
    scales_samples = np.array([1, 1, 1, 1, 1, 1, 1/250, 1/250, 1/250, 1/250, 1/160, 1/160])

    dataset_sensors = ParquetDataset(
        files, columns=feats_sensors, sample_length=sample_length,
        standardise=True, resample_dt=0.5
    )
    dataset_samples = ParquetDataset(
        files, columns=feats_samples, sample_length=sample_length,
        standardise=True,
    )

    # Get single sample
    dataloader = DataLoader(dataset_sensors, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        if i != sample_idx:
            continue
        
        with torch.no_grad():
            x, samples, ts = prepare_batch(
                batch, dataset_samples, n_samples=-1, ordered=True, device=device
            )
            predictions = model(x)
        
        # Move to CPU for plotting
        ts_np = ts.squeeze().cpu().numpy()
        pred_np = predictions.squeeze().cpu().numpy()
        true_np = samples.squeeze().cpu().numpy()
        
        # Create subplot grid for all features
        n_features = pred_np.shape[-1]
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = axes.flatten()
        
        for feature_idx in range(n_features):
            ax = axes[feature_idx]
            ax.plot(ts_np, true_np[:, feature_idx], label='Ground truth', alpha=0.8)
            ax.plot(ts_np, pred_np[:, feature_idx], label='Prediction', alpha=0.8, linestyle='--')
            
            title = feats_samples[feature_idx] if feature_idx < len(feats_samples) else f"Feature {feature_idx}"
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value (scaled)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f"MIONet Predictions (Sample {sample_idx})", fontsize=14)
        fig.tight_layout()
        
        return fig
    
    raise IndexError(f"Sample index {sample_idx} out of range")


if __name__ == "__main__":
    # Example usage
    checkpoint = r"runs\prototypes\deepOnet\testing\2026-02-19_09-41-18\checkpoints\checkpoint_epoch_20.pth"
    model = model_cnn_2()
    fig = plot_prediction(checkpoint, model)
    plt.show()