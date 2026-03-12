# jax_parquet_dataset.py

from __future__ import annotations

import json
import math
import threading
from queue import Queue
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import jax
import jax.numpy as jnp


class JAXParquetDataset:
    """
    Parquet-backed time series dataset optimized for JAX/Diffrax/Equinox training.

    Returns batches as:
      - t: (T,) jnp.float32  (shared time vector across the batch)
      - x: (B, T, F) jnp.float32  (states standardized)
      - meta_list: list[dict]  (Python-side metadata for bookkeeping; don't pass into jit)

    Key notes for JAX:
      - Data transfer to device happens via jax.device_put inside the iterator.
      - Batches are prefetched on a background thread to overlap I/O and compute.
      - Shapes are static across steps for stable jit compilation.

    Args:
        files: Parquet file paths.
        columns: Columns to load. 'time' will be prepended if missing.
        sample_length: Number of timesteps per sample window (before resampling).
        resample_every: Integer step for downsampling (keep every k-th row).
        resample_dt: Target dt; uses dt from file to compute step ≈ round(resample_dt / base_dt).
        standardise: Whether to standardize features.
        standardise_dict: Optional {'mean': np.ndarray, 'std': np.ndarray}.
        meta_key: Parquet schema metadata key containing JSON.
        cache_in_ram: If True, load each file once and keep arrays in RAM.
        dtype: jnp dtype for returned arrays.
    """

    def __init__(
        self,
        files: Sequence[Path],
        columns: Optional[List[str]] = None,
        sample_length: Optional[int] = None,
        resample_every: Optional[int] = None,
        resample_dt: Optional[float] = None,
        standardise: bool = True,
        standardise_dict: Optional[Dict[str, np.ndarray]] = None,
        meta_key: str = "run_params",
        cache_in_ram: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if len(files) == 0:
            raise AttributeError("File list must include at least one entry")

        self.files: List[Path] = list(files)
        self.dtype = dtype
        self.cache_in_ram = cache_in_ram

        # Normalize columns: ensure 'time' is first, preserve order
        if not columns:
            self.columns = None  # load all columns, but we'll reorder if needed
        else:
            cols = list(columns)
            if "time" in cols:
                cols.remove("time")
            self.columns = ["time"] + cols

        if resample_every is not None and resample_dt is not None:
            raise ValueError("Provide only one of resample_every or resample_dt")
        self.resample_every = resample_every
        self.resample_dt = resample_dt

        # Read a small sample to determine lengths & features
        first = self._read_parquet(self.files[0], columns=self.columns)
        series_length = first.shape[0]
        self.sample_length = sample_length if sample_length else series_length

        # How many non-overlapping windows per series (before resampling)
        self.n_per_series = int(math.floor(series_length / self.sample_length))
        if self.n_per_series <= 0:
            raise ValueError(
                f"Sample length {self.sample_length} exceeds series length {series_length}"
            )

        # Determine feature count and standardized statistics
        n_feats = first.shape[1] - 1  # exclude time
        if standardise:
            if standardise_dict is not None:
                self.standardise = {
                    "mean": np.asarray(standardise_dict["mean"], dtype=np.float32),
                    "std": np.asarray(standardise_dict["std"], dtype=np.float32),
                }
            else:
                self.standardise = self._compute_statistics()
        else:
            self.standardise = {
                "mean": np.zeros(n_feats, dtype=np.float32),
                "std": np.ones(n_feats, dtype=np.float32),
            }

        # Extract metadata blob per file
        self.metas: List[Dict[str, Any]] = []
        for f in self.files:
            schema = pq.read_schema(f)
            meta = schema.metadata or {}
            key_b = meta_key.encode("utf-8")
            if key_b in meta:
                params = json.loads(meta[key_b].decode("utf-8"))
            else:
                params = {}
            self.metas.append(params)

        # Optional: cache each file as a contiguous float32 array [T, 1+F]
        self._cache: List[Optional[np.ndarray]] = [None] * len(self.files)
        if cache_in_ram:
            for i, f in enumerate(self.files):
                arr = self._read_parquet(f, columns=self.columns)
                self._cache[i] = np.ascontiguousarray(arr, dtype=np.float32)

    # ----------------------------
    # Public API
    # ----------------------------
    def __len__(self) -> int:
        return self.n_per_series * len(self.files)
    
    @property
    def steps_per_epoch(self) -> int:
        """
        Number of steps (batches) per epoch, given the current dataset length and typical batch size.
        Usage: dataset.steps_per_epoch(batch_size)
        """
        # Default batch size is not known here, so return a function
        def get_steps(batch_size: int, drop_last: bool = False) -> int:
            total = len(self)
            n_full = total // batch_size
            if drop_last:
                return n_full
            else:
                return n_full + (1 if total % batch_size else 0)
        return get_steps
    
    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        prefetch: int = 2,
        key: Optional[jax.Array] = None,
        device: Optional[Any] = None,
    ) -> Iterator[Tuple[jax.Array, jax.Array, List[Dict[str, Any]]]]:
        """
        Yield batches (t, x, meta_list) where:
          - t: (T,) jnp.float32 on device
          - x: (T, B, F) jnp.float32 on device
          - meta_list: Python list of dicts for the B examples

        Notes:
          - meta_list is kept on host; don't pass it to jitted functions.
          - If you want deterministic shuffling, pass rng_key.
        """
        all_indices = self._build_indices()
        if shuffle:
            if key is None:
                rng = np.random.default_rng()
                rng.shuffle(all_indices)
            else:
                # Shuffle with JAX PRNG to be fully deterministic across runs
                perm = jax.random.permutation(key, len(all_indices))
                all_indices = [all_indices[i] for i in np.asarray(perm)]

        # Batch the indices
        total = len(all_indices)
        n_full = total // batch_size
        n_batches = n_full if drop_last else (n_full + (1 if total % batch_size else 0))

        def batch_iter() -> Iterator[Tuple[jax.Array, jax.Array, List[Dict[str, Any]]]]:
            for b in range(n_batches):
                start = b * batch_size
                stop = min((b + 1) * batch_size, total)
                batch_idx = all_indices[start:stop]
                if len(batch_idx) < batch_size and drop_last:
                    continue

                t, x, meta_list = self._load_batch(batch_idx)

                # Device transfer (async-friendly for GPU)
                if device is None:
                    t = jax.device_put(t)
                    x = jax.device_put(x)
                else:
                    t = jax.device_put(t, device=device)
                    x = jax.device_put(x, device=device)
                yield t, x, meta_list

        # Prefetch batches on a background thread to overlap IO/compute
        return _prefetch(batch_iter(), buffer_size=prefetch)

    def inverse_scale(self, states: jax.Array) -> jax.Array:
        """
        Undo standardization on (T, B, F) or (..., F) tensors.
        """
        mean = jnp.asarray(self.standardise["mean"], dtype=states.dtype)
        std = jnp.asarray(self.standardise["std"], dtype=states.dtype)
        return states * std + mean

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _build_indices(self) -> List[Tuple[int, int]]:
        """
        Build (file_idx, window_idx) pairs. Window is non-overlapping chunk:
          start = window_idx * sample_length, stop = start + sample_length
        """
        idx: List[Tuple[int, int]] = []
        for fi in range(len(self.files)):
            for wi in range(self.n_per_series):
                idx.append((fi, wi))
        return idx

    def _load_batch(
        self, batch_idx: Sequence[Tuple[int, int]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[Dict[str, Any]]]:
        xs_list: List[np.ndarray] = []
        ts_list: List[np.ndarray] = []
        meta_list: List[Dict[str, Any]] = []

        for (file_i, win_i) in batch_idx:
            t_np, x_np = self._get_window(file_i, win_i)
            xs_list.append(x_np)  # (T, F)
            ts_list.append(t_np)  # (T,)
            m = self.metas[file_i].copy()
            m["file_index"] = file_i
            m["window_index"] = win_i
            meta_list.append(m)

        # Validate aligned timesteps
        t0 = ts_list[0]
        for ti in ts_list[1:]:
            if not np.allclose(ti, t0, atol=5e-3):
                raise ValueError(
                    "Timesteps between runs are different. Please ensure they are equal"
                )

        # Stack -> (B, T, F)
        x_btf = np.stack(xs_list, axis=0) 
        t = jnp.asarray(t0, dtype=self.dtype)
        x = jnp.asarray(x_btf, dtype=self.dtype)
        return t, x, meta_list

    def _get_window(self, file_idx: int, window_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        arr = self._maybe_read(file_idx)  # (N, 1+F), float32
        start = window_idx * self.sample_length
        stop = (window_idx + 1) * self.sample_length
        states_time = arr[start:stop, :]  # (T, 1+F)

        # Optional resampling
        if self.resample_every is not None or self.resample_dt is not None:
            if states_time.shape[0] < 2:
                raise ValueError("Series must have at least two timesteps for resampling")
            if self.resample_every is not None:
                step = int(self.resample_every)
            else:
                base_dt = float(states_time[1, 0] - states_time[0, 0])
                if base_dt <= 0:
                    raise ValueError("Base timestep must be positive for resampling")
                step = int(round(float(self.resample_dt) / base_dt))
            if step <= 0:
                raise ValueError("Resample step must be >= 1")
            states_time = states_time[::step]

        # Standardize (x - mean) / std
        t = states_time[:, 0]
        x = states_time[:, 1:]
        x = (x - self.standardise["mean"]) / self.standardise["std"]

        # Make time start at zero (like your code) and round
        t = np.round(t - t[0], 2).astype(np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)
        return t, x

    def _maybe_read(self, file_idx: int) -> np.ndarray:
        if self.cache_in_ram:
            cached = self._cache[file_idx]
            assert cached is not None
            return cached
        return self._read_parquet(self.files[file_idx], columns=self.columns)

    def _read_parquet(self, path: Path, columns: Optional[List[str]]) -> np.ndarray:
        # Use pandas for simplicity, order columns with 'time' first if needed
        df = pd.read_parquet(path, columns=columns)
        vals = df.values
        if columns is None:
            # Ensure 'time' is first if present in arbitrary column order
            cols = list(df.columns)
            if "time" in cols and cols[0] != "time":
                time_idx = cols.index("time")
                perm = [time_idx] + [i for i in range(len(cols)) if i != time_idx]
                vals = vals[:, perm]
        return np.ascontiguousarray(vals, dtype=np.float32)

    def _compute_statistics(self) -> Dict[str, np.ndarray]:
        """
        Welford's algorithm over features (excluding time), equivalent to your approach.
        """
        n = 0
        # Determine feature count
        sample = self._maybe_read(0) if self.cache_in_ram else self._read_parquet(self.files[0], self.columns)
        n_feats = sample.shape[1] - 1
        mean = np.zeros(n_feats, dtype=np.float64)
        m2 = np.zeros(n_feats, dtype=np.float64)

        def update(arr_f32: np.ndarray):
            nonlocal n, mean, m2
            x = arr_f32[:, 1:].astype(np.float64, copy=False)  # exclude time
            for row in x:
                n += 1
                delta = row - mean
                mean += delta / n
                delta2 = row - mean
                m2 += delta * delta2

        if self.cache_in_ram:
            for cached in self._cache:
                if cached is None:
                    continue
                update(cached)
        else:
            for f in self.files:
                arr = self._read_parquet(f, columns=self.columns)
                update(arr)

        std = np.sqrt(m2 / max(n, 1)).astype(np.float32)
        std[std < 1e-8] = 1.0
        return {"mean": mean.astype(np.float32), "std": std}


# ----------------------------
# Prefetch utility
# ----------------------------
def _prefetch(
    it: Iterable[Tuple[jax.Array, jax.Array, List[Dict[str, Any]]]],
    buffer_size: int = 2,
) -> Iterator[Tuple[jax.Array, jax.Array, List[Dict[str, Any]]]]:
    """
    Simple threaded prefetcher. Pulls items from `it` on a background thread
    and fills a bounded queue. Useful to overlap Parquet I/O with device compute.
    """
    q: Queue = Queue(maxsize=max(1, buffer_size))
    sentinel = object()

    def _worker():
        try:
            for item in it:
                q.put(item)
        finally:
            q.put(sentinel)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item
