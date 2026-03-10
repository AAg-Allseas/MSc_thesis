import subprocess
import os

if __name__ == "__main__":
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        # List inside the Volume
        files = dbutils.fs.ls("/Volumes/main_udev/ai_labs/aag/artifacts/.internal")

        # Pick the newest wheel
        latest_dbfs = max(
            files, key=lambda f: f.modificationTime
        ).path  # e.g., "dbfs:/Volumes/.../thesis-1.0.4-..."
        latest_local = latest_dbfs.replace(
            "dbfs:", ""
        )  # -> "/Volumes/.../thesis-1.0.4-..."

        print(f"Installing: {latest_local}")
        subprocess.run(["pip", "install", latest_local], check=True)

from thesis.prototyping.dataloader import ParquetDataset
from thesis.prototyping.data_handling import find_parquet_files
from thesis.performance_tests.diffrax_model import LatentSDE
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax  # https://github.com/deepmind/optax
import mlflow

import torch
from pathlib import Path
from torch.utils.data import DataLoader


mlflow.enable_system_metrics_logging()


@eqx.filter_jit
def loss_fn(
    sde: LatentSDE, xs: jnp.ndarray, ts: jnp.ndarray, *, key, kl_weight: float = 1.0
) -> jnp.ndarray:
    """ELBO loss: -log p(x|z) + kl_weight * KL.

    Args:
        sde: LatentSDE model (single sample, no batch dim).
        xs: Observations, shape (T, data_size).
        ts: Timestamps, shape (T,).
        key: PRNG key.
        kl_weight: Annealing weight for KL term (0 → 1 over training).
    Returns:
        Scalar loss (negative ELBO).
    """
    log_pxs, kl = sde(xs, ts, key=key)
    return -log_pxs + kl_weight * kl


@eqx.filter_jit
def batch_loss_fn(
    sde: LatentSDE,
    xs_batch: jnp.ndarray,
    ts: jnp.ndarray,
    *,
    key,
    kl_weight: float = 1.0,
) -> jnp.ndarray:
    """Mean loss over a batch. xs_batch shape: (batch, T, data_size)."""
    batch_size = xs_batch.shape[0]
    keys = jr.split(key, batch_size)
    losses = jax.vmap(lambda x, k: loss_fn(sde, x, ts, key=k, kl_weight=kl_weight))(
        xs_batch, keys
    )
    return jnp.mean(losses)


def increase_update_initial(updates, sde):
    """Multiply gradient updates for initial condition params (pz0_mean, pz0_logvar, qz0_posterior) by 10."""
    initial_leaves = lambda u: [
        u.pz0_mean,
        u.pz0_logvar,
        u.qz0_posterior.weight,
        u.qz0_posterior.bias,
    ]
    return eqx.tree_at(initial_leaves, updates, replace_fn=lambda x: x * 10)


@eqx.filter_jit
def train_step(
    sde: LatentSDE, opt_state, optimizer, xs_batch, ts, *, key, kl_weight: float = 1.0
):
    loss, grads = eqx.filter_value_and_grad(batch_loss_fn)(
        sde, xs_batch, ts, key=key, kl_weight=kl_weight
    )
    grads = increase_update_initial(grads, sde)
    updates, opt_state = optimizer.update(grads, opt_state, sde)
    sde = eqx.apply_updates(sde, updates)
    return sde, opt_state, loss


if __name__ == "__main__":
    # --- Hyperparameters ---
    import os

    DATA_SIZE = 12
    LATENT_SIZE = 8  # compress 12→8: positions & velocities share structure
    CONTEXT_SIZE = 64
    HIDDEN_SIZE = 128
    LR_INIT = 1e-3  # conservative start; SDE training is sensitive
    LR_GAMMA = 0.999  # gentler decay (~0.6× after 500 epochs)
    NUM_EPOCHS = 2000  # start shorter, extend once loss stabilises
    KL_ANNEAL_ITERS = 500  # ramp KL over ~500 epochs to let decoder learn first
    BATCH_SIZE = 32  # moderate batch for stable gradients without OOM
    DT = 0.05
    SAMPLE_LENGTH = 5000  # 250 s windows — manageable for memory & gradients
    LOG_EVERY = 10
    SAMPLE_EVERY = 250
    CHECKPOINT_EVERY = 250  # save model every N epochs

    FEATS = [
        "pos_eta_x",
        "pos_eta_y",
        "pos_eta_mz",
        "pos_nu_x",
        "pos_nu_y",
        "pos_nu_mz",
        "rpm_bow_fore",
        "rpm_bow_aft",
        "rpm_stern_fore",
        "rpm_stern_aft",
        "rpm_fixed_ps",
        "rpm_fixed_sb",
    ]

    mlflow.set_tracking_uri("databricks")
    # Experiments go in Workspace, not Volumes
    mlflow.set_experiment(experiment_id="2984249850048933")
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        DATA_PATH = Path("/Volumes/main_udev/ai_labs/aag/data/pilot_3dof/")
        CHECKPOINT_DIR = Path(
            "/Volumes/main_udev/ai_labs/aag/artifacts/checkpoints/latentsde_diffrax"
        )
    else:
        DATA_PATH = Path(
            r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"
        )
        CHECKPOINT_DIR = Path("runs/latentSDE/checkpoints")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data loading ---
    files = find_parquet_files(
        DATA_PATH,
        lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05,
    )
    dataset = ParquetDataset(
        files, columns=FEATS, sample_length=SAMPLE_LENGTH, standardise=True
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(
        f"Loaded {len(files)} files, {len(dataset)} samples, {len(dataloader)} batches/epoch"
    )

    # --- Model + optimizer ---
    key = jax.random.key(7777)
    model_key, train_key = jr.split(key)

    sde = LatentSDE(DATA_SIZE, LATENT_SIZE, CONTEXT_SIZE, HIDDEN_SIZE, key=model_key)

    schedule = optax.exponential_decay(LR_INIT, transition_steps=1, decay_rate=LR_GAMMA)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    # --- Training loop ---
    import tempfile

    with mlflow.start_run(run_name=f"latentsde_diffrax_bs{BATCH_SIZE}_lr{LR_INIT}"):
        mlflow.log_params(
            {
                "model_type": "LatentSDE_diffrax",
                "batch_size": BATCH_SIZE,
                "latent_size": LATENT_SIZE,
                "context_size": CONTEXT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "lr_init": LR_INIT,
                "lr_gamma": LR_GAMMA,
                "num_epochs": NUM_EPOCHS,
                "kl_anneal_iters": KL_ANNEAL_ITERS,
                "dt": DT,
                "sample_length": SAMPLE_LENGTH,
                "n_files": len(files),
            }
        )

        global_step = 0
        best_loss = float("inf")

        for epoch in range(NUM_EPOCHS):
            epoch_key = jr.fold_in(train_key, epoch)
            kl_weight = min(1.0, epoch / KL_ANNEAL_ITERS)
            epoch_losses = []

            for i, (ts_batch, xs_batch, meta) in enumerate(dataloader):
                # Validate aligned timesteps across the batch
                t = ts_batch[0]
                if not torch.allclose(
                    ts_batch, t.unsqueeze(0).expand_as(ts_batch), atol=0.005
                ):
                    print(f"Skipping batch {i} — misaligned timesteps")
                    continue

                # Skip incomplete batches (last batch may be smaller)
                if xs_batch.shape[0] != BATCH_SIZE:
                    continue

                # Convert to JAX arrays: ts (T,), xs (batch, T, data_size)
                ts = jnp.asarray(t.numpy())
                xs = jnp.asarray(xs_batch.numpy())

                step_key = jr.fold_in(epoch_key, i)
                sde, opt_state, loss = train_step(
                    sde,
                    opt_state,
                    optimizer,
                    xs,
                    ts,
                    key=step_key,
                    kl_weight=kl_weight,
                )

                loss_val = float(loss)
                epoch_losses.append(loss_val)

                if global_step % LOG_EVERY == 0:
                    lr_now = float(schedule(global_step))
                    mlflow.log_metrics(
                        {
                            "batch_loss": loss_val,
                            "learning_rate": lr_now,
                            "kl_weight": kl_weight,
                        },
                        step=global_step,
                    )

                global_step += 1

            # Epoch summary
            if epoch_losses:
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                print(
                    f"Epoch {epoch:05d} | mean_loss: {mean_loss:.4f} | kl_w: {kl_weight:.3f}"
                )
                mlflow.log_metric("epoch_loss", mean_loss, step=epoch)

                # Save best model
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir) / f"best_{epoch:05d}.eqx"
                        eqx.tree_serialise_leaves(tmp_path, sde)
                        mlflow.log_metric("best_loss", best_loss, step=epoch)
                        mlflow.log_artifact(str(tmp_path), artifact_path="best")

            # Periodic checkpoint
            if epoch % CHECKPOINT_EVERY == 0 and epoch > 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / f"checkpoint_{epoch:05d}.eqx"
                    eqx.tree_serialise_leaves(tmp_path, sde)

                    mlflow.log_artifact(str(tmp_path), artifact_path="checkpoint")

            # Sample from prior and log figures
            if epoch % SAMPLE_EVERY == 0 and epoch > 0:
                sample_key = jr.fold_in(epoch_key, 999)
                sample_keys = jr.split(sample_key, 4)
                sample_ts = jnp.arange(0, SAMPLE_LENGTH * DT, DT)
                samples = jax.vmap(lambda k: sde.sample(sample_ts, key=k))(sample_keys)
                # samples: (4, T, data_size)

                fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
                for j, ax in enumerate(axes.flat):
                    for s in range(samples.shape[0]):
                        ax.plot(sample_ts, samples[s, :, j], alpha=0.6, linewidth=0.5)
                    ax.set_title(FEATS[j])
                fig.suptitle(f"Prior samples — Epoch {epoch}")
                plt.tight_layout()
                mlflow.log_figure(fig, f"samples/epoch_{epoch:05d}.png")
                plt.close(fig)

        # Save final model + log as artifact
        final_path = CHECKPOINT_DIR / "final.eqx"
        eqx.tree_serialise_leaves(final_path, sde)
        mlflow.log_artifact(str(final_path), artifact_path="checkpoints")
        mlflow.log_artifact(
            str(CHECKPOINT_DIR / "best.eqx"), artifact_path="checkpoints"
        )
        print(f"Training complete. Best loss: {best_loss:.4f}")
