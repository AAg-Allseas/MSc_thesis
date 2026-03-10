import subprocess
import os

# # Prevent JAX from pre-allocating all GPU memory, which starves PyTorch.
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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

from pathlib import Path

import numpy as np
from thesis.prototyping.data_handling import find_parquet_files
from thesis.prototyping.dataloader import ParquetDataset, prep_batch
from thesis.prototyping.latentSDE.test_latentSDE import load_and_sample
from thesis.toy_model.ornstein_uhlenbeck import (
    ou_generate_uniform,
    resample_from_base,
)
from thesis.toy_model.supply import SupplyVessel
from thesis.toy_model.gnc import attitudeEuler
from thesis.mss_model.simulation import SimConfig, simulate_osv
import torch
from torch.utils.data import DataLoader
import mlflow
import time
import gc


@mlflow.trace(name="toy_model")
def toy_model_simulation():
    """Benchmark the NumPy/CPU toy DP model over the same time horizon as the neural SDEs."""
    RUNTIME = 10800       # seconds – same as neural SDE benchmarks
    DT = 0.05
    N = int(RUNTIME // DT)
    SEED = 0
    MU_F = np.array([75e3, 0, 0])
    SIGMA_F = np.array([50e3, 0, 0])

    with mlflow.start_run(run_name="toy_model", nested=True):
        mlflow.log_params({
            "framework": "numpy",
            "model": "ToyDPModel",
            "runtime_s": RUNTIME,
            "dt": DT,
            "n_timesteps": N,
            "seed": SEED,
        })

        with mlflow.start_span(name="setup"):
            rng = np.random.default_rng(SEED)
            vessel = SupplyVessel("DPcontrol")

            noise_dt = 0.05
            f_ext = ou_generate_uniform(
                int(RUNTIME // noise_dt), noise_dt,
                mu=MU_F, sigma=SIGMA_F, rng=rng,
            )
            f_ext = resample_from_base(f_ext, noise_dt, DT, N * DT)

            eta = vessel.eta.copy()
            nu = vessel.nu.copy()
            u_actual = vessel.DPcontrol(eta, nu, DT)  # warm-start actuators

        with mlflow.start_span(name="simulation_loop"):
            # Store full trajectory like the neural SDEs do (216k × 12 features)
            trajectory = np.empty((N + 1, 12), dtype=np.float64)
            t_start = time.perf_counter()
            for i in range(N + 1):
                u_control = vessel.DPcontrol(eta, nu, DT)
                nu, u_actual = vessel.dynamics(
                    eta, nu, u_actual, u_control, DT, f_external=f_ext[i],
                )
                eta = attitudeEuler(eta, nu, DT)
                trajectory[i, :6] = eta
                trajectory[i, 6:] = u_actual
            t_elapsed = time.perf_counter() - t_start

        mlflow.log_metrics({
            "wall_time_s": t_elapsed,
            "timesteps_per_second": N / t_elapsed,
        })


@mlflow.trace(name="mss_model")
def mss_model_simulation():
    """Benchmark the 6-DOF MSS OSV model (RK4, optimal allocation, wave drift)."""
    RUNTIME = 10800
    DT = 0.05
    N = int(RUNTIME // DT)

    cfg = SimConfig(
        T_final=RUNTIME,
        h=DT,
        Vc=0.5,
        betaVc=np.deg2rad(-140),
        Hs=2.0,
        Tp=8.0,
        beta_wave=np.deg2rad(45),
        alloc_dynamic=True,
    )

    with mlflow.start_run(run_name="mss_model", nested=True):
        mlflow.log_params({
            "framework": "numpy",
            "model": "MSS_OSV_6DOF",
            "runtime_s": RUNTIME,
            "dt": DT,
            "n_timesteps": N,
            "integrator": "RK4",
            "alloc": "SLSQP_optimal",
            "Hs": cfg.Hs,
            "Tp": cfg.Tp,
            "beta_wave_deg": round(np.rad2deg(cfg.beta_wave), 1),
            "Vc": cfg.Vc,
        })

        with mlflow.start_span(name="simulation"):
            t_start = time.perf_counter()
            results = simulate_osv(cfg)
            t_elapsed = time.perf_counter() - t_start

        mlflow.log_metrics({
            "wall_time_s": t_elapsed,
            "timesteps_per_second": N / t_elapsed,
        })


@mlflow.trace(name="torchsde_prior")
def latent_sde_prior():
    BATCH_SIZE = 50
    END_TIME = 10800
    DT = 0.05

    with mlflow.start_run(run_name="torchsde_prior", nested=True):
        mlflow.log_params({
            "framework": "torchsde",
            "network": "prior",
            "batch_size": BATCH_SIZE,
            "end_time": END_TIME,
            "dt": DT,
            "n_timesteps": int(END_TIME / DT),
        })

        with mlflow.start_span(name="sde_integration"):
            t_start = time.perf_counter()
            with torch.no_grad():
                _ = load_and_sample(
                    Path("/Volumes/main_udev/ai_labs/aag/artifacts/latentsde_491.pth"),
                    batch_size=BATCH_SIZE,
                    data_size=12,
                    latent_size=4,
                    context_size=64,
                    hidden_size=128,
                    timestep=DT,
                    custom_sampling=False,
                    end_time=END_TIME,
                )
            del _
            t_elapsed = time.perf_counter() - t_start

        mlflow.log_metrics({
            "wall_time_s": t_elapsed,
            "samples_per_second": BATCH_SIZE / t_elapsed,
        })


@mlflow.trace(name="torchsde_posterior")
def latent_sde_posterior():
    BATCH_SIZE = 50
    END_TIME = 10800
    DT = 0.05

    with mlflow.start_span(name="data_loading"):
        files = find_parquet_files(
            Path("/Volumes/main_udev/ai_labs/aag/data/pilot_3dof/"),
            lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05,
        )

        feats = [
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

        dataset = ParquetDataset(
            files, columns=feats, sample_length=216000, standardise=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_states = []
        for i, batch in enumerate(dataloader):
            if i >= BATCH_SIZE:
                break
            _, states = prep_batch(batch, device=str(device))
            dataset_states.append(states)
            del _

    with mlflow.start_run(run_name="torchsde_posterior", nested=True):
        mlflow.log_params({
            "framework": "torchsde",
            "network": "posterior",
            "batch_size": BATCH_SIZE,
            "end_time": END_TIME,
            "dt": DT,
            "n_timesteps": int(END_TIME / DT),
        })

        batched_data_states = torch.cat(
            dataset_states, dim=1
        )  # (time_steps, batch_size, features)

        with mlflow.start_span(name="sde_integration"):
            t_start = time.perf_counter()
            with torch.no_grad():
                _ = load_and_sample(
                    Path("/Volumes/main_udev/ai_labs/aag/artifacts/latentsde_491.pth"),
                    batch_size=BATCH_SIZE,
                    data_size=12,
                    latent_size=4,
                    context_size=64,
                    hidden_size=128,
                    timestep=DT,
                    end_time=END_TIME,
                    network="posterior",
                    custom_sampling=False,
                    data_sample=batched_data_states,
                )
            del _, batched_data_states, dataset_states
            t_elapsed = time.perf_counter() - t_start

        mlflow.log_metrics({
            "wall_time_s": t_elapsed,
            "samples_per_second": BATCH_SIZE / t_elapsed,
        })

    # Clean up GPU objects that survive past the `del` above
    del dataset, dataloader


@mlflow.trace(name="diffrax_train_step")
def diffrax_sde_train_step():
    """Profile a single diffrax training step (forward + backward + optimizer update)."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import equinox as eqx

    from thesis.performance_tests.diffrax_model import LatentSDE
    from thesis.performance_tests.diffrax_train import train_step

    DATA_SIZE = 12
    LATENT_SIZE = 8
    CONTEXT_SIZE = 64
    HIDDEN_SIZE = 128
    BATCH_SIZE = 32
    SAMPLE_LENGTH = 5000
    DT = 0.05
    LR_INIT = 1e-3
    N_STEPS = 10

    key = jax.random.key(42)
    model_key, data_key, step_key = jr.split(key, 3)

    sde = LatentSDE(DATA_SIZE, LATENT_SIZE, CONTEXT_SIZE, HIDDEN_SIZE, key=model_key)

    schedule = optax.exponential_decay(LR_INIT, transition_steps=1, decay_rate=0.999)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    ts = jnp.arange(0, SAMPLE_LENGTH * DT, DT)
    xs_batch = jr.normal(data_key, (BATCH_SIZE, SAMPLE_LENGTH, DATA_SIZE))

    with mlflow.start_run(run_name="diffrax_train_step", nested=True):
        mlflow.log_params({
            "framework": "diffrax",
            "network": "train_step",
            "batch_size": BATCH_SIZE,
            "sample_length": SAMPLE_LENGTH,
            "dt": DT,
            "lr_init": LR_INIT,
            "n_steps": N_STEPS,
            "latent_size": LATENT_SIZE,
            "context_size": CONTEXT_SIZE,
            "hidden_size": HIDDEN_SIZE,
        })


        with mlflow.start_span(name="jit_compilation"):
            # Warmup: JIT compilation
            t_compile_start = time.perf_counter()
            sde, opt_state, loss = train_step(
                sde, opt_state, optimizer, xs_batch, ts, key=step_key, kl_weight=0.5
            )
            jax.block_until_ready(loss)
            t_compile = time.perf_counter() - t_compile_start
            mlflow.log_metric("jit_compile_s", t_compile)

        with mlflow.start_span(name="training_step"):
            # Timed run
            t_start = time.perf_counter()
            for i in range(N_STEPS):
                step_key = jr.fold_in(key, i + 100)
                sde, opt_state, loss = train_step(
                    sde, opt_state, optimizer, xs_batch, ts, key=step_key, kl_weight=0.5
                )
            jax.block_until_ready(loss)
            t_elapsed = time.perf_counter() - t_start

            mlflow.log_metrics({
                "wall_time_s": t_elapsed,
                "time_per_step_s": t_elapsed / N_STEPS,
                "samples_per_second": BATCH_SIZE * N_STEPS / t_elapsed,
                "final_loss": float(loss),
            })


@mlflow.trace(name="diffrax_prior_sample")
def diffrax_sde_sample():
    """Profile diffrax prior sampling (inference)."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from thesis.performance_tests.diffrax_model import LatentSDE

    DATA_SIZE = 12
    LATENT_SIZE = 8
    CONTEXT_SIZE = 64
    HIDDEN_SIZE = 128
    DT = 0.05
    END_TIME = 10800
    N_SAMPLES = 50

    key = jax.random.key(42)
    model_key, sample_key = jr.split(key)

    sde = LatentSDE(DATA_SIZE, LATENT_SIZE, CONTEXT_SIZE, HIDDEN_SIZE, key=model_key)
    ts = jnp.arange(0, END_TIME, DT)

    with mlflow.start_run(run_name="diffrax_prior_sample", nested=True):
        mlflow.log_params({
            "framework": "diffrax",
            "network": "prior",
            "n_samples": N_SAMPLES,
            "end_time": END_TIME,
            "dt": DT,
            "n_timesteps": int(END_TIME / DT),
            "latent_size": LATENT_SIZE,
            "context_size": CONTEXT_SIZE,
            "hidden_size": HIDDEN_SIZE,
        })

        # Warmup: JIT compilation
        with mlflow.start_span(name="jit_compilation"):
            t_compile_start = time.perf_counter()
            _ = sde.sample(ts, key=sample_key)
            jax.block_until_ready(_)
            t_compile = time.perf_counter() - t_compile_start
            mlflow.log_metric("jit_compile_s", t_compile)

        # Timed run
        sample_keys = jr.split(sample_key, N_SAMPLES)
        with mlflow.start_span(name="sde_integration"):
            t_start = time.perf_counter()
            samples = jax.vmap(lambda k: sde.sample(ts, key=k))(sample_keys)
            jax.block_until_ready(samples)
            t_elapsed = time.perf_counter() - t_start

            mlflow.log_metrics({
                "wall_time_s": t_elapsed,
                "time_per_sample_s": t_elapsed / N_SAMPLES,
                "samples_per_second": N_SAMPLES / t_elapsed,
            })


def _log_gpu(tag: str):
    """Log current GPU memory usage to MLflow."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    mlflow.log_metrics({
        f"gpu_allocated_gb_{tag}": round(alloc, 3),
        f"gpu_reserved_gb_{tag}": round(reserved, 3),
    })
    print(f"[GPU {tag}] allocated={alloc:.3f} GB  reserved={reserved:.3f} GB")


if __name__ == "__main__":
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id="3631831912551032")

    def _free_gpu(tag: str):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _log_gpu(tag)

    with mlflow.start_run(run_name="profiling"):
        # CPU-only models first (no GPU contention).
        toy_model_simulation()
        mss_model_simulation()

        # Run PyTorch first — JAX pre-allocates GPU memory that isn't
        # released, which can OOM the subsequent PyTorch runs.
        latent_sde_prior()
        _free_gpu("after_prior")
        latent_sde_posterior()
        _free_gpu("after_posterior")
        diffrax_sde_train_step()
        diffrax_sde_sample()
