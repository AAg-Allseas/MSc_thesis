from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def compare_timesteps() -> None:
    path = Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data\timestep_convergence")
    files = list(path.glob("*.parquet"))
    
    data_cols = None
    
    # Group files by seed and timestep
    file_info = {}
    for file in files:
        # Parse filename: dp_sim_duration_timestep_seed.parquet
        parts = file.stem.split("_")
        seed = int(parts[-1])
        timestep = float(parts[-2])
        
        if seed not in file_info:
            file_info[seed] = {}
        file_info[seed][timestep] = file
    
    # Find the baseline timestep (0.01)
    baseline_dt = 0.01
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    axs = axs.flatten()
    twinaxs = [ax.twinx() for ax in axs]
    
    # Dictionaries to store cumulative errors: {timestep: {col: [series_across_seeds]}}
    timestep_cumulative_series = {}
    timestep_times = {}
    
    # Color map for timesteps
    timesteps_set = set()
    for seed_files in file_info.values():
        timesteps_set.update(ts for ts in seed_files.keys() if ts != baseline_dt)
    sorted_timesteps = sorted(timesteps_set)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_timesteps)))
    timestep_colors = dict(zip(sorted_timesteps, colors))
    
    # Process each seed separately
    for seed, seed_files in file_info.items():
        if baseline_dt not in seed_files:
            continue
            
        baseline = pd.read_parquet(seed_files[baseline_dt], columns=["time", "pos_eta_x", "pos_eta_y", "pos_eta_mz", 'rpm_bow_fore', 'rpm_bow_aft', 'rpm_stern_fore', 'rpm_stern_aft', 'rpm_fixed_ps', 'rpm_fixed_sb'])
        
        if data_cols is None:
            data_cols = [col for col in baseline.columns if col != "time"]
        
        for dt, file in seed_files.items():
            if dt == baseline_dt:
                continue
                
            df = pd.read_parquet(file, columns=["time", "pos_eta_x", "pos_eta_y", "pos_eta_mz", 'rpm_bow_fore', 'rpm_bow_aft', 'rpm_stern_fore', 'rpm_stern_aft', 'rpm_fixed_ps', 'rpm_fixed_sb'])
            
            # Merge on time to align baseline and df
            merged = pd.merge(baseline, df, on="time", suffixes=("_baseline", "_df"))
            
            # Initialize storage for this timestep
            if dt not in timestep_cumulative_series:
                timestep_cumulative_series[dt] = {col: [] for col in data_cols}
            
            # Calculate error for each column
            for idx, col in enumerate(data_cols):
                col_baseline = f"{col}_baseline"
                col_df = f"{col}_df"
                if col_baseline in merged.columns and col_df in merged.columns:
                    merged[f"{col}_error"] = np.abs(merged[col_df] - merged[col_baseline])
                    merged[f"{col}_cumulative_error"] = (merged[f"{col}_error"] * merged["time"].diff().fillna(0)).cumsum()
                    
                    # Store cumulative error series for this seed
                    timestep_cumulative_series[dt][col].append(merged[f"{col}_cumulative_error"].to_numpy())
                    if dt not in timestep_times:
                        timestep_times[dt] = []
                    timestep_times[dt].append(merged["time"].to_numpy())
                    
                    # Plot point error on primary axis
                    ax = axs[idx]
                    ax.plot(merged["time"], merged[f"{col}_error"], 
                           color=timestep_colors[dt], alpha=0.3, linewidth=0.5)
    
    # Calculate average cumulative errors over time and plot them with error bars
    avg_cumulative_errors = {}
    handles = []
    labels = []
    for dt in sorted_timesteps:
        avg_cumulative_errors[dt] = {}
        for idx, col in enumerate(data_cols):
            series_list = timestep_cumulative_series[dt][col]
            if not series_list:
                continue
            min_len = min(len(series) for series in series_list)
            trimmed_stack = np.vstack([series[:min_len] for series in series_list])
            mean_series = trimmed_stack.mean(axis=0)
            std_series = trimmed_stack.std(axis=0)
            avg_cumulative_errors[dt][col] = {
                "mean": mean_series,
                "std": std_series,
                "final_mean": mean_series[-1],
                "final_std": std_series[-1],
            }

            ax2 = twinaxs[idx]
            times = timestep_times[dt][0][:min_len]
            stride = max(1, len(times) // 50)
            ax2.plot(
                times,
                mean_series,
                linestyle="--",
                color=timestep_colors[dt],
                linewidth=1.5,
                alpha=0.9,
            )
            ax2.fill_between(
                times,
                mean_series - std_series,
                mean_series + std_series,
                color=timestep_colors[dt],
                alpha=0.15,
            )

            if idx == 0:
                handles.append(
                    plt.Line2D([0], [0], linestyle="--", color=timestep_colors[dt], linewidth=2)
                )
                labels.append(f"dt={dt:.2f}")
    
    for idx, col in enumerate(data_cols):
        ax = axs[idx]
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{col} Error")
        ax.set_title(f"{col}")
        
        ax2 = twinaxs[idx]
        ax2.set_ylabel("Cumulative Error")
    
    # Add shared legend
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(sorted_timesteps))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Create second figure for final cumulative error vs timestep
    fig2, axs2 = plt.subplots(3, 3, figsize=(12, 10))
    axs2 = axs2.flatten()
    
    for idx, col in enumerate(data_cols):
        ax = axs2[idx]
        
        mean_errors = [avg_cumulative_errors[dt][col]['final_mean'] for dt in sorted_timesteps]
        std_errors = [avg_cumulative_errors[dt][col]['final_std'] for dt in sorted_timesteps]
        
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)
        ax.plot(sorted_timesteps, mean_errors, marker='o', linewidth=2)
        ax.fill_between(
            sorted_timesteps,
            mean_errors - std_errors,
            mean_errors + std_errors,
            alpha=0.2,
        )
        ax.set_xlabel("Timestep (s)")
        ax.set_ylabel("Final Cumulative Error")
        ax.set_title(f"{col}")
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(sorted_timesteps))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

        

if __name__ == "__main__":
    compare_timesteps()