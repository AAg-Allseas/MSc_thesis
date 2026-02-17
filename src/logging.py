"""Custom logging configuration for training experiments.

Provides custom log levels (BATCH, EPOCH) and pre-configured loggers that
write to multiple rotating files for easy filtering and debugging. Also
includes utilities for saving model checkpoints and tracking losses.

Example:
    Basic training loop with logging, loss tracking, and checkpointing::

        from src.logging import (
            create_run_folder, get_logger, LossTracker, save_checkpoint
        )

        # Set up run folder and logger
        run_path = create_run_folder("runs", "experiment_name")
        logger = get_logger(run_path)
        tracker = LossTracker(run_path)

        for epoch in range(n_epochs):
            for batch in dataloader:
                loss = train_step(model, batch)
                tracker.log_batch(loss.item())
                logger.batch("Batch loss: %.4f", loss.item())

            mean_loss = tracker.end_epoch(epoch)
            logger.epoch("Epoch %d complete, mean loss: %.4f", epoch, mean_loss)
            save_checkpoint(model, run_path, epoch, optimizer, scheduler)

    The resulting folder structure::

        runs/experiment_name/2026-02-17_14-30-00/
            checkpoints/
                checkpoint_epoch_0.pth
                checkpoint_epoch_1.pth
            logs/
                training.log   # all messages
                batch.log      # BATCH level only
                epoch.log      # EPOCH level only
                warnings.log   # WARNING and above
                errors.log     # ERROR and above
                losses.json    # batch and epoch loss history
"""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import torch
from torch import nn

# ---- Custom log levels ----
BATCH_LEVEL_NUM = 15   # between DEBUG (10) and INFO (20)
EPOCH_LEVEL_NUM = 25   # between INFO (20) and WARNING (30)

logging.addLevelName(BATCH_LEVEL_NUM, "BATCH")
logging.addLevelName(EPOCH_LEVEL_NUM, "EPOCH")

def batch(self: logging.Logger, message: str, *args, **kws) -> None:
    """Log a message at BATCH level.

    Args:
        message: The log message (may contain %-style format specifiers).
        *args: Arguments to merge into message via %-formatting.
        **kws: Keyword arguments passed to Logger._log.
    """
    if self.isEnabledFor(BATCH_LEVEL_NUM):
        self._log(BATCH_LEVEL_NUM, message, args, **kws)


def epoch(self: logging.Logger, message: str, *args, **kws) -> None:
    """Log a message at EPOCH level.

    Args:
        message: The log message (may contain %-style format specifiers).
        *args: Arguments to merge into message via %-formatting.
        **kws: Keyword arguments passed to Logger._log.
    """
    if self.isEnabledFor(EPOCH_LEVEL_NUM):
        self._log(EPOCH_LEVEL_NUM, message, args, **kws)

logging.Logger.batch = batch
logging.Logger.epoch = epoch


# ---- Simple filters to route records by level ----
class LevelOnlyFilter(logging.Filter):
    """Allow only log records matching an exact level.

    Attributes:
        level_num: The numeric log level to accept.
    """

    def __init__(self, level_num: int) -> None:
        """Initialize the filter.

        Args:
            level_num: The exact numeric log level to pass through.
        """
        super().__init__()
        self.level_num = level_num

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if the record should be logged.

        Args:
            record: The log record to evaluate.

        Returns:
            True if the record's level matches exactly, False otherwise.
        """
        return record.levelno == self.level_num


class LevelAtLeastFilter(logging.Filter):
    """Allow log records at or above a minimum level.

    Attributes:
        min_level_num: The minimum numeric log level to accept.
    """

    def __init__(self, min_level_num: int) -> None:
        """Initialize the filter.

        Args:
            min_level_num: The minimum numeric log level to pass through.
        """
        super().__init__()
        self.min_level_num = min_level_num

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if the record should be logged.

        Args:
            record: The log record to evaluate.

        Returns:
            True if the record's level is at or above the threshold.
        """
        return record.levelno >= self.min_level_num


# ---- Run folder utilities ----
def create_run_folder(base: str = "runs", sub: Optional[str] = None) -> Path:
    """Create a timestamped run folder with a logs subdirectory.

    Args:
        base: Base directory for all runs.
        sub: Unused placeholder for future subfolders.

    Returns:
        Path to the newly created run folder.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if sub:
        base += f"/{sub}" 
    run_path = Path(base) / ts
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    return run_path


# ---- Formatter ----
def _make_formatter() -> logging.Formatter:
    """Create the standard formatter for file handlers."""
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# ---- Handler builder helpers ----
def _rotating_file_handler(
    path: Path,
    level: int,
    *,
    max_bytes: int = 2_000_000,
    backup_count: int = 3,
    formatter: logging.Formatter | None = None,
    flt: logging.Filter | None = None,
) -> RotatingFileHandler:
    """Build a rotating file handler with optional filter."""
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    handler.setLevel(level)
    if formatter is None:
        formatter = _make_formatter()
    handler.setFormatter(formatter)
    if flt is not None:
        handler.addFilter(flt)
    return handler


def _console_handler(
    level: int,
    formatter: logging.Formatter | None = None,
    flt: logging.Filter | None = None,
) -> logging.StreamHandler:
    """Build a console (stderr) handler with optional filter."""
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if formatter is None:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    if flt is not None:
        ch.addFilter(flt)
    return ch


# ---- Public API ----
def get_logger(
    run_path: Path,
    name: str = "train",
    console_level: int = EPOCH_LEVEL_NUM,
) -> logging.Logger:
    """Create or retrieve a configured logger for a training run.

    Sets up multiple rotating file handlers:
        - training.log: all levels
        - batch.log: BATCH level only
        - epoch.log: EPOCH level only
        - warnings.log: WARNING and above
        - errors.log: ERROR and above
        - Console: EPOCH and above (configurable)

    Idempotent: safe to call multiple times without duplicating handlers.

    Args:
        run_path: Directory for log files (a 'logs' subfolder will be created).
        name: Logger name (allows multiple independent loggers).
        console_level: Minimum level for console output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # capture everything; handlers will filter

    # Prevent duplicate handlers if called multiple times
    # We detect if we've already configured this logger for the given run_path
    sentinel_attr = "_configured_for_run"
    if getattr(logger, sentinel_attr, None) == str(run_path):
        return logger

    log_dir = run_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = _make_formatter()

    # All-in-one log
    all_handler = _rotating_file_handler(
        path=log_dir / "training.log",
        level=logging.DEBUG,
        formatter=fmt
    )

    # Exact-level logs
    batch_handler = _rotating_file_handler(
        path=log_dir / "batch.log",
        level=BATCH_LEVEL_NUM,
        formatter=fmt,
        flt=LevelAtLeastFilter(BATCH_LEVEL_NUM)
    )
    epoch_handler = _rotating_file_handler(
        path=log_dir / "epoch.log",
        level=EPOCH_LEVEL_NUM,
        formatter=fmt,
        flt=LevelAtLeastFilter(EPOCH_LEVEL_NUM)
    )

    # Threshold logs
    warnings_handler = _rotating_file_handler(
        path=log_dir / "warnings.log",
        level=logging.WARNING,
        formatter=fmt,
        flt=LevelAtLeastFilter(logging.WARNING)
    )
    errors_handler = _rotating_file_handler(
        path=log_dir / "errors.log",
        level=logging.ERROR,
        formatter=fmt,
        flt=LevelAtLeastFilter(logging.ERROR)
    )

    # Console (keep concise by default)
    console = _console_handler(level=console_level)

    # Attach
    for h in (all_handler, batch_handler, epoch_handler, warnings_handler, errors_handler, console):
        logger.addHandler(h)

    # Reduce noisy third-party libs if needed (optional)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Mark configured for this run path
    setattr(logger, sentinel_attr, str(run_path))
    return logger


def save_checkpoint(
    model: nn.Module,
    run_path: Path,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Save model checkpoint to the run folder.

    Saves model weights and optionally optimizer/scheduler state to
    `run_path/checkpoints/checkpoint_epoch_{epoch}.pth`.

    Args:
        model: The model to save.
        run_path: Root directory of the training run.
        epoch: Current epoch number (used in filename).
        optimizer: Optional optimizer to save state from.
        scheduler: Optional scheduler to save state from.
        extra: Optional dict of additional items to include.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, path)
    return path


class LossTracker:
    """Track and persist batch/epoch losses during training.

    Accumulates batch losses within an epoch, computes the mean at epoch end,
    and saves all metrics to a JSON file in `run_path/logs/losses.json`.

    Attributes:
        run_path: Root directory of the training run.
        batch_losses: List of batch losses for the current epoch.
        epoch_history: List of dicts with 'epoch', 'mean_loss', and 'batch_losses'.
    """

    def __init__(self, run_path: Path) -> None:
        """Initialize the loss tracker.

        Args:
            run_path: Root directory of the training run.
        """
        self.run_path = run_path
        self.batch_losses: list[float] = []
        self.epoch_history: list[dict[str, Any]] = []
        self._losses_file = run_path / "logs" / "losses.json"
        self._losses_file.parent.mkdir(parents=True, exist_ok=True)

    def log_batch(self, loss: float) -> None:
        """Record a single batch loss.

        Args:
            loss: The loss value for this batch.
        """
        self.batch_losses.append(loss)

    def end_epoch(self, epoch: int) -> float:
        """Finalize the current epoch, compute mean, and save to disk.

        Args:
            epoch: The epoch number that just completed.

        Returns:
            Mean loss for the epoch.
        """
        mean_loss = sum(self.batch_losses) / len(self.batch_losses) if self.batch_losses else 0.0
        self.epoch_history.append({
            "epoch": epoch,
            "mean_loss": mean_loss,
            "batch_losses": self.batch_losses.copy(),
        })
        self._save()
        self.batch_losses.clear()
        return mean_loss

    def _save(self) -> None:
        """Write epoch history to JSON file."""
        with open(self._losses_file, "w", encoding="utf-8") as f:
            json.dump(self.epoch_history, f, indent=2)