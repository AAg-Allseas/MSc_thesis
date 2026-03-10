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
from thesis.prototyping.dataloader import ParquetDataset
from thesis.prototyping.data_handling import find_parquet_files

mlflow.enable_system_metrics_logging()
