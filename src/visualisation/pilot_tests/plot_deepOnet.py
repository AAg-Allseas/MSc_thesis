
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
from src.prototyping.deepOnet.models import model_1dof, model_1dof_2, model_2, model_cnn_1, model_cnn_2
from src.prototyping.deepOnet.utils import BranchConstructor, MLPConstructor, prepare_batch
from src.visualisation.general_plotting.config import LINEWIDTH



def plot_prediction(
    checkpoint_path: str | Path,
    model: MIONet,
    data_path: Optional[str | Path] = None,
    sample_idx: int = 0,
    device: Optional[torch.device] = None,
    max_plot_points: int = 2000,
) -> plt.Figure:
    """Plot model prediction vs ground truth for all features.

    Args:
        checkpoint_path: Path to the saved checkpoint (.pth file).
        data_path: Path to data directory. Defaults to standard location.
        sample_idx: Index of sample in the dataset to plot.
        device: Torch device. Defaults to CUDA if available.
        max_plot_points: Maximum number of points to plot (downsamples if needed).

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
    feats_sensors = ['tau_ext_x']
    feats_samples = [
        'pos_eta_x'
    ]


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
                batch, dataset_samples, n_samples=-1, ordered=True, device=device, input_features = {"surge_force": 0}
            )
            predictions = model(x)
        
        predictions = dataset_samples.inverse_scale(predictions)
        samples = dataset_samples.inverse_scale(samples)
        # Move to CPU for plotting
        ts_np = ts.squeeze().cpu().numpy() * 500 # Rescale
        pred_np = predictions.squeeze().cpu().numpy()
        true_np = samples.squeeze().cpu().numpy()
    
        # Downsample if necessary to avoid memory issues
        n_points = len(ts_np)
        if n_points > max_plot_points:
            step = n_points // max_plot_points
            ts_np = ts_np[::step]
            pred_np = pred_np[::step]
            true_np = true_np[::step]
        
        # Ensure arrays are 2D (add feature dimension if needed)
        if pred_np.ndim == 1:
            pred_np = pred_np[:, np.newaxis]
        if true_np.ndim == 1:
            true_np = true_np[:, np.newaxis]
        
        # Create subplot grid for all features
        n_features = pred_np.shape[-1]
        # Compute grid as square as possible
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(0.75 * LINEWIDTH, 0.75 * 3/4 * LINEWIDTH))
        if n_features > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for feature_idx in range(n_features):
            ax = axes[feature_idx]
            ax.plot(ts_np, true_np[:, feature_idx], label='Ground truth', alpha=0.8)
            ax.plot(ts_np, pred_np[:, feature_idx], label='Prediction', alpha=0.8, linestyle='--')
            
            title = feats_samples[feature_idx] if feature_idx < len(feats_samples) else f"Feature {feature_idx}"
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value (scaled)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        fig.tight_layout()
        
        return fig
    
    raise IndexError(f"Sample index {sample_idx} out of range")


def main():
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Example usage
    checkpoint = r"C:/Soft_dev/MSc_thesis/mlruns/2/0e12a6785b6445a289a629b3ad998da3/artifacts/checkpoints/checkpoint_epoch_1350.pth"
    model = model_1dof_2()
    return plot_prediction(checkpoint, model, sample_idx=10)

if __name__ == "__main__":
    # Use non-interactive backend to avoid tkinter memory issues
    main()
    plt.show()