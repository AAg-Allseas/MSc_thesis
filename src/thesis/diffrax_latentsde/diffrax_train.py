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

from thesis.diffrax_latentsde.vector_fields import FieldConfig, FieldType
from thesis.prototyping.dataloader import ParquetDataset
from thesis.prototyping.data_handling import find_parquet_files
from thesis.diffrax_latentsde.diffrax_model import LatentSDE
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import mlflow
import tempfile

import torch
from pathlib import Path
from torch.utils.data import DataLoader


mlflow.enable_system_metrics_logging()


@eqx.filter_jit
def loss_fn(
    sde: LatentSDE, xs: jnp.ndarray, ts: jnp.ndarray, *, key, kl_weight: float = 1.0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ELBO loss: -log p(x|z) + kl_weight * KL.

    Args:
        sde: LatentSDE model (single sample, no batch dim).
        xs: Observations, shape (T, data_size).
        ts: Timestamps, shape (T,).
        key: PRNG key.
        kl_weight: Annealing weight for KL term (0 → 1 over training).
    Returns:
        Scalar loss (negative ELBO).
        kl_initial: Initial KL divergence.
        kl_path: KL divergence accumulated along the SDE path.
        log_pxs: Log likelihood of data given latent path.
    """
    log_pxs, kl_initial, kl_path = sde(xs, ts, key=key)
    return -log_pxs + kl_weight * (kl_initial + kl_path), kl_initial, kl_path, log_pxs


@eqx.filter_jit
def batch_loss_fn(
    sde: LatentSDE,
    xs_batch: jnp.ndarray,
    ts: jnp.ndarray,
    *,
    key,
    kl_weight: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Mean loss over a batch. xs_batch shape: (batch, T, data_size)."""
    batch_size = xs_batch.shape[0]
    keys = jr.split(key, batch_size)
    # losses shape: (batch, 4) if loss_fn returns 4 values
    losses, kl_initials, kl_paths, log_pxs = jax.vmap(
        lambda x, k: loss_fn(sde, x, ts, key=k, kl_weight=kl_weight)
    )(xs_batch, keys)
    # Average each component separately for logging
    return (
        jnp.mean(losses),
        (jnp.mean(kl_initials),
        jnp.mean(kl_paths),
        jnp.mean(log_pxs))
    )


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
    # batch_loss_fn returns (elbo, kl_initial, kl_path, log_pxs)
    (elbo, (kl_initial, kl_path, log_pxs)), grads = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)(
        sde, xs_batch, ts, key=key, kl_weight=kl_weight
    )
    grads = increase_update_initial(grads, sde)
    updates, opt_state = optimizer.update(grads, opt_state, sde)
    sde = eqx.apply_updates(sde, updates)
    return sde, opt_state, elbo, kl_initial, kl_path, log_pxs


def main(data_path: Path, checkpoint_path: Path) -> None:
    # --- Hyperparameters ---
    # Model params
    DATA_SIZE = 12
    LATENT_SIZE = 24  
    CONTEXT_SIZE = 64
    HIDDEN_SIZE = 128
    DEPTH = 4
    CONTROL_SIZE = 3 # Dimension of noise control for g field (e.g., 3 for 3D Brownian motion)
    f_type = FieldType.CONTEXT_STATE
    h_type = FieldType.STATE
    g_type = FieldType.STATE

    # Training params
    LR_INIT = 1e-3 
    NUM_EPOCHS = 1000
    KL_ANNEAL_ITERS = 200
    BATCH_SIZE = 32
    DT = 0.2
    SAMPLE_LENGTH = 5000 
    LOG_EVERY = 10
    SAMPLE_EVERY = 250
    CHECKPOINT_EVERY = 250


    # --- Model ---
    key = jax.random.key(7777)
    model_key, train_key = jr.split(key)

    f_key, h_key, g_key, model_key = jr.split(model_key, 4)

    # Field parameters
    f_config = FieldConfig(
        field_type=f_type,
        latent_size=LATENT_SIZE,
        context_size=CONTEXT_SIZE,
        hidden_layer_width=HIDDEN_SIZE,
        depth=DEPTH,
        key=f_key,
    )
    h_config = FieldConfig(
        field_type = h_type,
        latent_size=LATENT_SIZE,
        hidden_layer_width=HIDDEN_SIZE,
        depth=DEPTH,
        key=h_key,
    )
    g_config = FieldConfig(
        field_type = g_type,
        latent_size=LATENT_SIZE,
        hidden_layer_width=HIDDEN_SIZE,
        control_size=CONTROL_SIZE,
        depth=DEPTH,
        key=g_key,
    )

    sde = LatentSDE(DATA_SIZE, LATENT_SIZE, CONTEXT_SIZE, HIDDEN_SIZE, f_config, h_config, g_config, key=model_key)

    # --- Data ---
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



    checkpoint_path.mkdir(parents=True, exist_ok=True)


    files = find_parquet_files(
        data_path,
        lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05,
    )
    dataset = ParquetDataset(
        files, columns=FEATS, sample_length=SAMPLE_LENGTH, standardise=True
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(
        f"Loaded {len(files)} files, {len(dataset)} samples, {len(dataloader)} batches/epoch"
    )


    # --- Optimiser ---
    schedule = optax.schedules.cosine_onecycle_schedule(transition_steps = NUM_EPOCHS * len(dataloader), peak_value=LR_INIT, pct_start=0.1)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))


    # --- Training loop ---
    with mlflow.start_run(run_name=f"latentsde_diffrax_bs{BATCH_SIZE}_lr{LR_INIT}_"):
        mlflow.log_params(
            {
                "model_type": "LatentSDE_diffrax",
                "batch_size": BATCH_SIZE,
                "latent_size": LATENT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "f_type": f_type.value,
                "context_size": CONTEXT_SIZE,
                "h_type": h_type.value,
                "g_type": g_type.value,
                "control_size": CONTROL_SIZE,
                "lr_init": LR_INIT,
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
                sde, opt_state, elbo, kl_initial, kl_path, log_pxs = train_step(
                    sde,
                    opt_state,
                    optimizer,
                    xs,
                    ts,
                    key=step_key,
                    kl_weight=kl_weight,
                )

                elbo_val = float(elbo)
                kl_initial_val = float(kl_initial)
                kl_path_val = float(kl_path)
                log_pxs_val = float(log_pxs)
                epoch_losses.append(elbo_val)

                if global_step % LOG_EVERY == 0:
                    lr_now = float(schedule(global_step))
                    mlflow.log_metrics(
                        {
                            "batch_loss": elbo_val,
                            "kl_initial": kl_initial_val,
                            "kl_path": kl_path_val,
                            "log_pxs": log_pxs_val,
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
        final_path = checkpoint_path / "final.eqx"
        eqx.tree_serialise_leaves(final_path, sde)
        mlflow.log_artifact(str(final_path), artifact_path="checkpoints")
        mlflow.log_artifact(
            str(checkpoint_path / "best.eqx"), artifact_path="checkpoints"
        )
        print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
 
    mlflow.set_experiment("diffrax_latentsde_local")
    data_path = Path(
        r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"
    )
    checkpoint_path = Path("runs/latentSDE/checkpoints")

    main(data_path, checkpoint_path)