# Databricks Cheatsheet — MLflow Experiments for PyTorch Training

> **Docs**: [MLflow Tracking](https://docs.databricks.com/en/mlflow/tracking.html) ·
> [Log runs](https://docs.databricks.com/en/mlflow/tracking-runs.html) ·
> [MLflow Python API](https://mlflow.org/docs/latest/python_api/mlflow.html) ·
> [PyTorch integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html) ·
> [Search syntax](https://mlflow.org/docs/latest/search-runs.html)

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Python Environment Setup (Cluster & Serverless)](#2-python-environment-setup)
3. [MLflow Setup](#3-mlflow-setup)
4. [Creating & Setting Experiments](#4-creating--setting-experiments)
5. [Logging Parameters](#5-logging-parameters)
6. [Logging Metrics (Losses, LR, etc.)](#6-logging-metrics)
7. [Logging Artifacts (Checkpoints, Plots, Data)](#7-logging-artifacts)
8. [Logging Models](#8-logging-models)
9. [Tags & Run Metadata](#9-tags--run-metadata)
10. [Nested Runs (Hyperparameter Sweeps)](#10-nested-runs)
11. [Autologging](#11-autologging)
12. [Resuming / Continuing Runs](#12-resuming--continuing-runs)
13. [Querying & Comparing Runs](#13-querying--comparing-runs)
14. [Loading Models & Artifacts from Runs](#14-loading-models--artifacts-from-runs)
15. [Full Training Loop Example (MIONet-style)](#15-full-training-loop-example)
16. [Full Training Loop Example (LatentSDE-style)](#16-full-training-loop-example-latentsde)
17. [Tips & Gotchas](#17-tips--gotchas)

---

## 1. Core Concepts

Databricks uses **MLflow** as its experiment tracking backbone. MLflow is already integrated into every Databricks workspace — there is nothing extra to install. The mental model is straightforward: you organise work into **Experiments**, and every time you train a model that creates a **Run** inside the experiment. Everything about that run (hyperparameters, loss curves, saved weights, plots) lives together and is browsable in the Databricks UI.

| Concept        | What it is |
|----------------|-----------|
| **Experiment** | A named container that groups related runs (e.g. `deepOnet_v2`). Maps to a folder in the Databricks workspace. Think of it like a project folder. |
| **Run**        | A single training execution inside an experiment. Stores params, metrics, artifacts, and model. Each run gets a unique ID. |
| **Parameter**  | A single key-value pair logged **once** per run (hyperparams, architecture choices). Immutable after logging — useful for recording the configuration of a run. |
| **Metric**     | A numeric value logged at successive **steps** (loss, lr). Step-indexed metrics are plotted automatically as line charts in the UI, making it easy to compare convergence across runs. |
| **Artifact**   | Any file attached to a run (checkpoints, plots, config JSONs, numpy arrays). Stored in Databricks-managed cloud storage — no need to wrangle S3/ADLS paths yourself. |
| **Model**      | A logged PyTorch model with a signature, ready for serving or reloading. MLflow wraps it so it can be loaded back with a single call. |
| **Tag**        | Free-form key-value metadata for filtering and searching runs (e.g. `model_type=MIONet`). |

The typical workflow is:

1. **Set an experiment** — tells MLflow where to file your runs.
2. **Start a run** — opens a tracking context.
3. **Log params** — record hyperparameters so you can reproduce the run later.
4. **Log metrics in your training loop** — loss, learning rate, KL weight, etc.
5. **Log artifacts / model** — save checkpoints, plots, the final model.
6. **End the run** — happens automatically with a `with` block.
7. **Compare** — use the Databricks UI or the search API to find your best run.

---

## 2. Python Environment Setup

Before you can train, Databricks needs to have the right packages available. Your project depends on packages like `torch`, `torchsde`, `torchvision`, `pyarrow`, `pandas`, `tqdm`, etc. (see `pyproject.toml`). Since you don't have admin access to modify the cluster configuration directly, you install dependencies from inside your code — either via `%pip` in a notebook or via a subprocess call in a `.py` script.

Both classic clusters and serverless compute support this workflow. The approach is the same in both cases. The key difference is that serverless **only** supports this approach (no init scripts, no cluster UI library installs), while classic clusters also support other methods you may not have access to.

> **Docs**: [Databricks Libraries](https://docs.databricks.com/en/libraries/index.html) ·
> [Serverless notebook dependencies](https://docs.databricks.com/en/serverless-compute/serverless-notebook-dependencies.html) ·
> [%pip in notebooks](https://docs.databricks.com/en/libraries/notebooks-python-libraries.html)

### 2a. Generating a requirements file from `pyproject.toml` (local, one-time)

Since this project uses **uv**, you can export a lockfile or a pip-compatible requirements file locally. This is what you'll upload alongside your code so that Databricks can install the same packages.

```powershell
# From the project root on your laptop
uv pip compile pyproject.toml -o requirements.txt
```

This produces a `requirements.txt` with pinned versions that pip (on Databricks) can consume directly. Commit it to your repo so it's available in Databricks Repos.

> **Note on PyTorch CUDA wheels**: Your `pyproject.toml` configures a custom index for `torch` and `torchvision` (`pytorch-cu128`). Databricks GPU runtimes (Runtime ML) come with a compatible PyTorch + CUDA already installed, so you generally **don't** need to reinstall torch on Databricks — just install the extra dependencies (`torchsde`, etc.). If the pre-installed torch version doesn't match what you need, you can override it with `--index-url` (see below).

### 2b. Installing dependencies — Notebook workflow

This is the most common way to work on Databricks when you don't control the cluster. Run `%pip` in the first cell of your notebook. This installs packages into the notebook's Python process without needing cluster admin access.

```python
# Cell 1 — Install dependencies
# Skip torch/torchvision if the cluster's pre-installed version is fine
%pip install torchsde tqdm matplotlib pyarrow pandas numpy scikit-learn

# IMPORTANT: restart the Python process so the new packages are importable
dbutils.library.restartPython()
```

> **Why `restartPython()`?** Databricks caches imported modules. After a `%pip install`, the interpreter must restart to pick up newly installed packages. Without this call you may get `ModuleNotFoundError` even though pip reports success.

To install from your committed requirements file:

```python
%pip install -r /Workspace/Repos/<user>/MSc_thesis/requirements.txt
dbutils.library.restartPython()
```

If you need a specific torch version (e.g. the cluster's pre-installed version is too old for `torchsde`):

```python
%pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
%pip install torchsde torchvision
dbutils.library.restartPython()
```

### 2c. Installing dependencies — Python script workflow

If you're running a `.py` script (e.g. via a Databricks Job or `dbutils.notebook.run()`), you can't use the `%pip` magic. Instead, install packages at the top of your script using `subprocess`. This has the same effect as `%pip`.

```python
import subprocess
import sys

def install_requirements():
    """Install project dependencies if not already available."""
    try:
        import torchsde  # Check if key dependency is present
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "torchsde", "tqdm", "matplotlib", "pyarrow",
            "pandas", "numpy", "scikit-learn",
        ])

install_requirements()
```

Or install from a requirements file:

```python
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", "/Workspace/Repos/<user>/MSc_thesis/requirements.txt",
])
```

> **Important**: Unlike notebooks, there is no `restartPython()` equivalent in a `.py` script. Packages installed via `subprocess` are available immediately to subsequent `import` statements *in the same process*, so this works as long as you install before you import.

### 2d. Uploading your project code

Your training scripts import from `src.prototyping.*`. To make these importable on Databricks:

1. **Databricks Repos (Git integration)** — link your GitHub/Azure DevOps repo. The repo is cloned to `/Workspace/Repos/<user>/MSc_thesis/`. You can then `import src.prototyping.dataloader` etc. directly. This is the cleanest approach.
   ```python
   # Add repository root to sys.path so `src.*` imports work
   import sys
   sys.path.insert(0, "/Workspace/Repos/<user>/MSc_thesis")
   ```

2. **Upload as a wheel** — build your project locally with `uv build`, then install on Databricks:
   ```powershell
   # Locally
   uv build
   ```
   ```python
   # On Databricks
   %pip install /Workspace/Repos/<user>/MSc_thesis/dist/msc_thesis_awn_aperghis-1.0.0-py3-none-any.whl
   dbutils.library.restartPython()
   ```

### 2e. Verifying the environment

After installation, verify everything is available. This works in both notebooks and scripts:

```python
import torch
import torchsde
import pyarrow
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"torchsde: {torchsde.__version__}")
print(f"PyArrow: {pyarrow.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 2f. GPU-specific notes

Databricks clusters with GPU runtimes (e.g. **Databricks Runtime ML** with `Standard_NC6s_v3` on Azure or `g4dn.xlarge` on AWS) come with PyTorch, CUDA, and cuDNN **pre-installed**. Check what's there before installing anything:

```python
import torch
print(torch.__version__)         # e.g. 2.1.0+cu121
print(torch.cuda.is_available()) # True
```

If the pre-installed version is compatible with your code, you only need to install the extra packages (`torchsde`, etc.) — this saves time and avoids CUDA version conflicts.

### 2g. Serverless GPU limitations

As of early 2026, serverless compute with GPU support is in **limited availability** (check your workspace's region). If GPU serverless is not available, use a classic GPU cluster. Serverless CPU-only nodes are generally available.

### 2h. Quick-start: first cells of every training notebook

```python
# Cell 1 — Install dependencies (skip torch if pre-installed version is fine)
%pip install -r /Workspace/Repos/<user>/MSc_thesis/requirements.txt
dbutils.library.restartPython()
```

```python
# Cell 2 — Verify environment and set up imports
import sys
sys.path.insert(0, "/Workspace/Repos/<user>/MSc_thesis")

import torch, torchsde
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")

import mlflow
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
```

---

## 3. MLflow Setup & Local Development

### On Databricks

MLflow is pre-installed on every Databricks runtime — there is nothing to `pip install`. The tracking URI is automatically configured to point at your workspace's MLflow backend, so logged runs appear in the Databricks UI immediately.

```python
import mlflow
import mlflow.pytorch

# Verify connection (should print "databricks")
print(mlflow.get_tracking_uri())
```

### Running locally (on your laptop)

**Yes, your code with MLflow logging will work locally.** MLflow is a pure-Python library that runs anywhere — it does not require Databricks. When you run locally, MLflow defaults to logging everything to a **local `./mlruns/` folder** in your working directory instead of the Databricks backend. All the same API calls (`log_params`, `log_metric`, `log_artifact`, etc.) work identically.

This means you can develop and test your training scripts on your laptop without changing any MLflow code. The runs just end up in a local folder instead of the Databricks UI.

#### Install MLflow locally (with uv)

```powershell
# From your project root
uv add mlflow
```

Or install it alongside your project:

```powershell
uv pip install mlflow
```

#### Local tracking: what happens by default

When you don't set a tracking URI, MLflow uses a file-based backend:

```python
import mlflow
print(mlflow.get_tracking_uri())  # prints something like "file:///C:/Soft_dev/MSc_thesis/mlruns"
```

All runs are stored in `./mlruns/` as flat files. You can view them in the MLflow UI:

```powershell
# Launch the local MLflow UI (opens in browser at http://localhost:5000)
uv run mlflow ui
```

This gives you the same experiment/run/metric/artifact browsing experience as the Databricks UI, just running locally.

#### Pointing local runs at Databricks (optional)

If you want local training runs to appear in the Databricks workspace, configure the tracking URI to point at Databricks. This requires authentication:

1. Install the Databricks CLI and configure a profile:
   ```powershell
   uv pip install databricks-cli
   databricks configure --token
   # Enter your workspace URL (e.g. https://adb-1234567890.azuredatabricks.net)
   # Enter a personal access token
   ```

2. Set the tracking URI in your script:
   ```python
   import mlflow
   mlflow.set_tracking_uri("databricks")
   # Now all log_* calls go to the Databricks backend
   ```

> **When to do this**: Only if you want to log local development runs to the shared Databricks experiment. For quick local testing, the default local `./mlruns/` backend is simpler — no authentication needed.

#### Writing code that works in both environments

Your training code doesn't need any `if/else` logic. The same `mlflow.log_*` calls work everywhere. The only difference is *where* the data ends up:

| Where you run         | Tracking URI       | Runs stored in              |
|-----------------------|--------------------|-----------------------------|
| Databricks notebook   | `databricks` (auto)| Databricks workspace        |
| Databricks job        | `databricks` (auto)| Databricks workspace        |
| Local laptop (default)| `file:///...`      | `./mlruns/` folder          |
| Local → Databricks    | `databricks` (set) | Databricks workspace        |

```python
# This code works unchanged in ALL environments:
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="test"):
    mlflow.log_params({"lr": 0.001, "epochs": 100})
    for epoch in range(100):
        loss = train_one_epoch(...)
        mlflow.log_metric("loss", loss, step=epoch)
    mlflow.pytorch.log_model(model, "model")
```

---

## 4. Creating & Setting Experiments

Before you start logging anything, you need to tell MLflow *which experiment* to file runs under. Call `set_experiment()` once at the top of your script or notebook. If the experiment doesn't exist yet, it's created automatically.

```python
# Set experiment by workspace path (created automatically if it doesn't exist).
# Using a path under /Users/<your-email>/ keeps it in your personal folder.
mlflow.set_experiment("/Users/<your-email>/experiments/deepOnet_v2")

# Or set by a short name (ends up at the workspace root).
mlflow.set_experiment("deepOnet_v2")
```

> **Docs**: [mlflow.set_experiment](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment)

Every `mlflow.start_run()` call after this will create a new run inside that experiment. You only need to call `set_experiment` once per notebook/script — it stays active for the rest of the session.

---

## 5. Logging Parameters

Parameters capture the **configuration** of a run — things that are fixed before training starts and don't change. Log them once at the beginning of each run. This is what lets you later answer questions like "which learning rate gave the best result?" by filtering and sorting in the UI.

Use `log_param()` for a single value or `log_params()` to log a whole dict at once.

```python
with mlflow.start_run(run_name="mionet_lr1e-3_bs16"):
    # Single param
    mlflow.log_param("latent_dim", 128)

    # Bulk params (dict)
    mlflow.log_params({
        "batch_size": 16,
        "n_samples": 2048,
        "n_epochs": 10000,
        "lr_init": 1e-3,
        "warmup_epochs": 100,
        "sample_length": 10000,
        "sample_dt": 0.5,
        "loss_fn": "MSELoss",
        "optimizer": "Adam",
        "grad_clip": 1.0,
        "output_dim": 12,
        "method": "euler",
        "features_sensors": str(feats_sensors),
        "features_samples": str(feats_samples),
        "n_training_files": len(files_training),
        "n_testing_files": len(files_testing),
    })
```

> **Limits**: param values are strings, max 500 chars. For larger config blobs, log as an artifact (see §6).

---

## 6. Logging Metrics

Metrics are the numbers that change during training — loss, learning rate, KL divergence weight, etc. Unlike parameters, you log the same metric key many times with an increasing `step` value. This creates a time series that the Databricks UI renders as a line chart, making it easy to see convergence or spot divergence.

The function signature is `log_metric(key, value, step)`. The `step` argument is what creates the x-axis in the UI plots. Use a **global batch counter** for batch-level metrics, and the **epoch number** for epoch-level summaries.

### Per-batch metrics

```python
global_step = 0

for epoch in range(n_epochs):
    for batch in dataloader:
        loss = train_step(...)

        mlflow.log_metric("batch_train_loss", loss.item(), step=global_step)
        global_step += 1
```

### Per-epoch metrics

```python
    mean_train_loss, mean_test_loss = loss_tracker.end_epoch(epoch)

    mlflow.log_metrics({
        "epoch_train_loss": mean_train_loss,
        "epoch_test_loss": mean_test_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }, step=epoch)
```

### Multiple metrics at once

```python
mlflow.log_metrics({
    "log_pxs": log_pxs.item(),
    "log_ratio": log_ratio.item(),
    "kl_weight": kl_scheduler.val,
    "total_loss": loss.item(),
}, step=epoch)
```

> **Docs**: [mlflow.log_metric](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric) ·
> [mlflow.log_metrics](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metrics)

---

## 7. Logging Artifacts

Artifacts are arbitrary files that you attach to a run. They're stored in Databricks-managed cloud storage and are downloadable from the UI or the API at any time. Typical uses:

- **Model checkpoints** (`.pth` files) so you can resume training or load the best epoch.
- **Loss histories** (numpy arrays, JSON) for post-hoc analysis or custom plots.
- **Figures** (PNG, SVG) for a visual summary saved alongside the run.
- **Config files** (JSON, YAML) capturing the full experiment setup.

The workflow is always: write the file to local (temp) disk first, then call `log_artifact()` or `log_artifacts()` to upload it. Databricks handles the cloud storage path for you.

### Single file

```python
# Save checkpoint to local disk, then log
torch.save(model.state_dict(), "/tmp/checkpoint_epoch_50.pth")
mlflow.log_artifact("/tmp/checkpoint_epoch_50.pth", artifact_path="checkpoints")
```

### Entire directory

```python
# Log all files in a directory
mlflow.log_artifacts("/tmp/my_plots/", artifact_path="plots")
```

### Numpy arrays / JSON config

```python
import numpy as np, json, tempfile, os

# Save numpy losses and log
with tempfile.TemporaryDirectory() as tmp:
    np.save(os.path.join(tmp, "kl_losses.npy"), kl_losses)
    np.save(os.path.join(tmp, "likelihood_losses.npy"), likelihood_losses)

    config = {"feats": feats, "scales": scales.tolist()}
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    mlflow.log_artifacts(tmp, artifact_path="data")
```

### Matplotlib figures

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(epoch_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

mlflow.log_figure(fig, "plots/loss_curve.png")
plt.close(fig)
```

> **Docs**: [mlflow.log_artifact](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact) ·
> [mlflow.log_figure](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_figure)

---

## 8. Logging Models

MLflow has first-class support for logging entire PyTorch models (not just state dicts). When you call `mlflow.pytorch.log_model()`, MLflow serialises the model, records its class, and creates a standardised "MLmodel" wrapper. This means you can reload the model later with a single call — no need to reconstruct the architecture manually. You can also optionally **register** the model in the Databricks Model Registry for versioning and deployment.

### Log at end of training

```python
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    registered_model_name="MIONet_v2",   # optional: register in Model Registry
)
```

### Log with input example (enables signature inference)

```python
import numpy as np

# Create a sample input matching your model's forward() signature
sample_input = (
    {
        "initial_conditions": torch.randn(1, 12),
        "surge_force": torch.randn(1, 500),
        "sway_force": torch.randn(1, 500),
        "yaw_moment": torch.randn(1, 500),
    },
    torch.randn(1, 100),
)

mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    input_example=sample_input,
)
```

### Load a model back

```python
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
loaded_model.eval()
```

> **Docs**: [mlflow.pytorch.log_model](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model)

---

## 9. Tags & Run Metadata

Tags are free-form key-value strings attached to a run. They're distinct from parameters — while parameters record the numeric/structural configuration, tags capture descriptive metadata like who ran it, what kind of model it is, or which dataset version was used. You can filter and search runs by tag in both the UI and the API, which is invaluable when you have dozens of runs in an experiment.

```python
mlflow.set_tags({
    "model_type": "MIONet",
    "dataset_version": "v2",
    "author": "AAg",
    "description": "Warmup + ReduceLROnPlateau, CNN branches",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
})
```

System tags (auto-set by Databricks):

| Tag | Description |
|-----|-------------|
| `mlflow.source.name` | Notebook/script path |
| `mlflow.runName` | Display name in UI |
| `mlflow.databricks.notebookPath` | Databricks notebook path |

---

## 10. Nested Runs (Hyperparameter Sweeps)

When you want to systematically try different hyperparameter combinations, you can create a **parent run** that contains multiple **child runs** (one per combination). This keeps the UI tidy — the parent acts as a collapsible folder, and you can compare the children side-by-side. The key is passing `nested=True` when starting a child run inside an already-active parent run.

```python
mlflow.set_experiment("/Users/<email>/experiments/latentSDE_sweep")

with mlflow.start_run(run_name="hyperparameter_sweep") as parent:
    for lr in [1e-3, 5e-3, 1e-2]:
        for hidden_size in [64, 128, 256]:
            with mlflow.start_run(run_name=f"lr={lr}_h={hidden_size}", nested=True):
                mlflow.log_params({
                    "lr_init": lr,
                    "hidden_size": hidden_size,
                    "latent_size": 4,
                    "context_size": 64,
                })

                # ... train model ...

                mlflow.log_metrics({
                    "final_loss": final_loss,
                    "final_log_pxs": log_pxs.item(),
                })
```

In the UI, child runs appear collapsible under the parent.

---

## 11. Autologging

MLflow offers an "autolog" feature that hooks into training frameworks and automatically captures metrics, parameters, and models without explicit `log_*` calls. For **PyTorch Lightning** users this works well out of the box. For vanilla `nn.Module` training loops (like yours), autologging has limited value — it can capture optimizer params and a model summary, but it can't automatically hook into your custom training step. **Manual logging is recommended** for your codebase because it gives you full control over what's tracked and at what granularity. That said, here's how to enable it:

```python
# Captures optimizer params, model summary, and gradients
mlflow.pytorch.autolog(
    log_every_n_epoch=1,      # Metric logging frequency
    log_models=True,          # Auto-log model at end
    log_datasets=False,       # Skip dataset logging
)
```

> **Docs**: [mlflow.pytorch.autolog](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.autolog)

---

## 12. Resuming / Continuing Runs

Sometimes training crashes, a cluster gets pre-empted, or you want to do a second training phase (e.g. fine-tuning) and log it under the same run. MLflow lets you reopen an existing run by passing its `run_id` to `start_run()`. All new metrics, artifacts, and tags are appended to the same run — the UI stitches the metric histories together seamlessly.

```python
# Get the run_id from a previous run
run_id = "abc123def456"

with mlflow.start_run(run_id=run_id):
    # Continue logging metrics from where you left off
    for epoch in range(restart_epoch, n_epochs):
        # ... train ...
        mlflow.log_metric("epoch_train_loss", mean_loss, step=epoch)
```

### Finding the last run in an experiment

```python
experiment = mlflow.get_experiment_by_name("/Users/<email>/experiments/deepOnet_v2")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
last_run_id = runs.iloc[0]["run_id"]
```

---

## 13. Querying & Comparing Runs

Once you have many runs, you'll want to find the best one or compare groups of runs. You can do this visually in the Databricks Experiments UI (select runs → click **Compare**), or programmatically with `mlflow.search_runs()`. The search API returns a Pandas DataFrame, so you can sort, filter, and analyse runs with familiar tools.

The filter syntax uses a SQL-like DSL: `metrics.<key>`, `params.<key>`, `tags.<key>`, and `attributes.<key>` (for built-in fields like `start_time` or `run_id`).

### Search runs programmatically

```python
import mlflow

experiment = mlflow.get_experiment_by_name("/Users/<email>/experiments/deepOnet_v2")

# All runs
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Filtered: only runs with test loss < 0.01
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.epoch_test_loss < 0.01",
    order_by=["metrics.epoch_test_loss ASC"],
)

# Filter by parameter
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.batch_size = '16' AND params.lr_init = '0.001'",
)

# Filter by tag
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.model_type = 'MIONet'",
)

print(runs_df[["run_id", "params.lr_init", "metrics.epoch_test_loss"]])
```

> **Docs**: [mlflow.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs) ·
> [Search syntax](https://mlflow.org/docs/latest/search-runs.html)

### In the Databricks UI

- Navigate to **Experiments** in the sidebar
- Click your experiment name
- Use the **Compare** button to select multiple runs for side-by-side charts

---

## 14. Loading Models & Artifacts from Runs

One of the biggest advantages of logging artifacts and models to MLflow is that you can retrieve them later from any notebook or cluster without hunting for file paths. MLflow provides two main mechanisms:

- `mlflow.artifacts.download_artifacts()` — downloads a specific artifact file (checkpoint, numpy array, etc.) to local disk and returns the local path.
- `mlflow.pytorch.load_model()` — loads a previously logged PyTorch model directly into memory, ready for inference or further training.

Both take a `run_id` to identify which run to pull from.

### Load a checkpoint artifact

```python
import mlflow, torch

run_id = "abc123"
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="checkpoints/checkpoint_epoch_50.pth",
)

checkpoint = torch.load(local_path, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

### Load a logged model

```python
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
model.eval()
```

### Load numpy artifacts

```python
import numpy as np

local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/kl_losses.npy",
)
kl_losses = np.load(local_path)
```

---

## 15. Full Training Loop Example

This example shows how to integrate MLflow into a complete training loop, adapted from the MIONet/DeepONet training script. The key idea is that the MLflow calls are lightweight additions around your existing training code — the model, data, and optimizer setup stay exactly the same.

```python
import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
import tempfile, os, numpy as np
import tqdm

# ── Experiment setup ─────────────────────────────────────────
mlflow.set_experiment("/Users/<email>/experiments/deepOnet_v2")

# ── Hyperparameters ──────────────────────────────────────────
latent_dim = 128
output_dim = 12
n_samples = 2048
batch_size = 16
n_epochs = 10000
lr_init = 1e-3
warmup_epochs = 100
max_errors = 3
sample_length = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build model, data, optimizer (your existing code) ────────
# model = MIONet(branches, trunk, output_dim).to(device)
# dataset_training_samples, dataloader_training = load_samples_sensors(...)
# dataset_testing_samples, dataloader_testing = load_samples_sensors(...)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
# scheduler = ...
# loss_fn = nn.MSELoss()

# ── MLflow Run ───────────────────────────────────────────────
with mlflow.start_run(run_name=f"mionet_bs{batch_size}_lr{lr_init}"):

    # Log params
    mlflow.log_params({
        "model_type": "MIONet",
        "latent_dim": latent_dim,
        "output_dim": output_dim,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "n_epochs": n_epochs,
        "lr_init": lr_init,
        "warmup_epochs": warmup_epochs,
        "sample_length": sample_length,
        "loss_fn": "MSELoss",
        "optimizer": "Adam",
        "grad_clip": 1.0,
        "n_training_files": len(files_training),
        "n_testing_files": len(files_testing),
    })
    mlflow.set_tags({
        "model_type": "MIONet",
        "author": "AAg",
    })

    global_step = 0

    for epoch in tqdm.tqdm(range(n_epochs)):
        model.train()
        for batch in dataloader_training:
            optimizer.zero_grad()
            x, samples, _ = prepare_batch(batch, dataset_training_samples, n_samples=n_samples, device=device)
            loss = loss_fn(model(x), samples)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Log batch loss
            mlflow.log_metric("batch_train_loss", loss.item(), step=global_step)
            global_step += 1

        # ── Evaluation ───────────────────────────────────────
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in dataloader_testing:
                x, samples, _ = prepare_batch(batch, dataset_testing_samples, n_samples=n_samples, device=device)
                test_loss = loss_fn(model(x), samples)
                if torch.isfinite(test_loss):
                    test_losses.append(test_loss.item())

        mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0.0

        # ── Log epoch metrics ────────────────────────────────
        lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metrics({
            "epoch_train_loss": loss.item(),  # or use your LossTracker mean
            "epoch_test_loss": mean_test_loss,
            "learning_rate": lr,
        }, step=epoch)

        # ── Checkpoint every 50 epochs ───────────────────────
        if epoch % 50 == 0:
            with tempfile.TemporaryDirectory() as tmp:
                ckpt_path = os.path.join(tmp, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        scheduler.step()

    # ── Log final model ──────────────────────────────────────
    mlflow.pytorch.log_model(model, artifact_path="model")
```

---

## 16. Full Training Loop Example (LatentSDE)

This example adapts the LatentSDE training script. The LatentSDE has a different loss structure (log-likelihood + KL divergence) and uses a KL annealing schedule, so we log more granular metrics. Notice that the MLflow integration pattern is identical — params at the start, metrics in the loop, artifacts periodically, model at the end.

```python
import mlflow
import mlflow.pytorch
import torch
import torchsde
import numpy as np
import tempfile, os
import tqdm

mlflow.set_experiment("/Users/<email>/experiments/latentSDE")

# ── Hyperparams ──────────────────────────────────────────────
batch_size = 25
latent_size = 4
context_size = 64
hidden_size = 128
lr_init = 5e-3
lr_gamma = 0.997
num_epochs = 5000
kl_anneal_iters = 1000
noise_std = 0.1
method = "euler"
dt = 0.05
sample_length = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build model, data, optimizer (your existing code) ────────
# latent_sde = LatentSDE(...).to(device)
# dataset = ParquetDataset(...)
# dataloader = DataLoader(dataset, ...)
# optimizer = optim.Adam(latent_sde.parameters(), lr=lr_init)
# scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
# kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

with mlflow.start_run(run_name=f"latentsde_h{hidden_size}_lr{lr_init}"):

    mlflow.log_params({
        "model_type": "LatentSDE",
        "batch_size": batch_size,
        "latent_size": latent_size,
        "context_size": context_size,
        "hidden_size": hidden_size,
        "lr_init": lr_init,
        "lr_gamma": lr_gamma,
        "num_epochs": num_epochs,
        "kl_anneal_iters": kl_anneal_iters,
        "noise_std": noise_std,
        "method": method,
        "dt": dt,
        "sample_length": sample_length,
        "n_files": len(files),
        "features": str(feats),
    })

    global_step = 0

    for epoch in tqdm.trange(num_epochs):
        for i, batch in enumerate(dataloader):
            ts, xs = prep_batch(batch, device)
            if xs.shape != (sample_length, batch_size, len(feats)):
                continue

            bm = torchsde.BrownianInterval(
                t0=ts[0], t1=ts[-1],
                size=(batch_size, latent_size + 1), dt=dt, device=device,
            )
            latent_sde.zero_grad()
            log_pxs, log_ratio = latent_sde(xs, ts, method=method, dt=dt, bm=bm)
            loss = -log_pxs + log_ratio * kl_scheduler.val

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_sde.parameters(), max_norm=1.0)
            optimizer.step()

            # Log per-batch
            mlflow.log_metrics({
                "batch_loss": loss.item(),
                "batch_log_pxs": log_pxs.item(),
                "batch_kl": log_ratio.item(),
            }, step=global_step)
            global_step += 1

        scheduler.step()
        kl_scheduler.step()

        # Log per-epoch
        lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metrics({
            "epoch_loss": loss.item(),
            "epoch_log_pxs": log_pxs.item(),
            "epoch_log_ratio": log_ratio.item(),
            "learning_rate": lr,
            "kl_weight": kl_scheduler.val,
        }, step=epoch)

        # Checkpoint
        if epoch % 50 == 0:
            with tempfile.TemporaryDirectory() as tmp:
                ckpt_path = os.path.join(tmp, f"latentsde_{epoch}.pth")
                torch.save(latent_sde.state_dict(), ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

                np.save(os.path.join(tmp, "kl_losses.npy"), kl_losses[:epoch+1])
                np.save(os.path.join(tmp, "likelihood_losses.npy"), likelihood_losses[:epoch+1])
                mlflow.log_artifacts(tmp, artifact_path="data")

    # Final model
    mlflow.pytorch.log_model(latent_sde, artifact_path="model")
```

---

## 17. Tips & Gotchas

### Performance

- **Don't log every batch if you have thousands per epoch.** Each `log_metric()` call is a network round-trip to the Databricks tracking server. If you're doing thousands of batches per epoch, this overhead adds up. Either log every N-th batch, or accumulate batch losses and only log epoch-level summaries.
  ```python
  if global_step % 100 == 0:
      mlflow.log_metric("batch_loss", loss.item(), step=global_step)
  ```

- **Batch your artifact uploads.** Write files to a temp directory and use `log_artifacts()` (plural) instead of calling `log_artifact()` many times.

### Context manager vs. manual start/end

```python
# Preferred: context manager (auto-ends run on exception too)
with mlflow.start_run():
    ...

# Manual (useful for notebooks)
run = mlflow.start_run()
# ... do work ...
mlflow.end_run()
```

### Get current run info

```python
run = mlflow.active_run()
print(run.info.run_id)
print(run.info.artifact_uri)
```

### System metrics (GPU utilization, memory)

Enable with:
```python
mlflow.enable_system_metrics_logging()
```
This logs GPU memory, CPU usage, etc. automatically.

> **Docs**: [System metrics](https://mlflow.org/docs/latest/system-metrics/index.html)

### Databricks-specific

- **Unity Catalog integration**: Register models to UC for governance.
  ```python
  mlflow.set_registry_uri("databricks-uc")
  mlflow.pytorch.log_model(model, artifact_path="model",
                            registered_model_name="catalog.schema.MIONet")
  ```
- **Cluster GPU selection**: Use GPU-enabled clusters (e.g. `Standard_NC6s_v3` on Azure, or `g4dn.xlarge` on AWS).
- **DBFS paths**: Artifacts stored automatically in the workspace. No need to manage DBFS paths manually.
- **Notebook widgets** for quick param sweeps:
  ```python
  dbutils.widgets.text("lr", "0.001")
  dbutils.widgets.text("batch_size", "16")
  lr = float(dbutils.widgets.get("lr"))
  batch_size = int(dbutils.widgets.get("batch_size"))
  ```

### Common search queries

```python
# Best run by test loss
best = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.epoch_test_loss ASC"],
    max_results=1,
)

# Runs from today
mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="attributes.start_time > '2026-02-20'",
)

# Runs with a specific tag
mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.model_type = 'LatentSDE'",
)
```

### Minimal quick-start template

```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="quick_test"):
    mlflow.log_params({"lr": 0.001, "epochs": 100})

    for epoch in range(100):
        loss = train_one_epoch(...)
        mlflow.log_metric("loss", loss, step=epoch)

    mlflow.pytorch.log_model(model, "model")
```

