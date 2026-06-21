"""Utility functions for UFM-Transformer project.

This module provides shared utility functions used across the UFM-Transformer
project, including device selection, checkpoint management, reproducibility
helpers, timing utilities, and parameter counting.

Classes:
    AverageMeter: Tracks running averages for metrics like loss and accuracy.
    Timer: Context manager for timing code blocks.
    Logger: Dual logging to file and console.

Functions:
    set_seed: Set random seeds for reproducibility across all backends.
    get_device: Auto-select the best available compute device.
    save_checkpoint: Save model checkpoint with full training state.
    load_checkpoint: Load model checkpoint, optionally resuming training state.
    count_parameters: Count trainable parameters in a PyTorch model.
    compute_flops: Estimate floating point operations for a model.
    print_system_info: Print system configuration and library versions.

Author: UFM-Transformer Team
"""

import logging
import os
import random
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all relevant libraries.

    This function seeds PyTorch (CPU and CUDA), NumPy, and Python's built-in
    ``random`` module. For CUDA backends, it also configures deterministic
    behaviour where possible, trading some performance for reproducibility.

    Args:
        seed: Integer seed value. Default is ``42``.

    Raises:
        ValueError: If ``seed`` is negative.

    Example:
        >>> set_seed(2024)
        >>> torch.randn(3)  # Now deterministic
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Seed all available CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms – may reduce throughput but improve
        # reproducibility.  Not all ops have deterministic impls, so we
        # warn rather than fail.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # MPS backend (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # PyTorch 2.x compile guard – deterministic compile is opt-in
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass

    logging.info("Random seed set to %d (deterministic mode enabled)", seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device(priority: str = "auto") -> torch.device:
    """Auto-select the best available compute device.

    Selection order:
        1. CUDA GPU (if available and ``priority`` is ``"auto"`` or ``"cuda"``)
        2. Apple MPS (if available and ``priority`` is ``"auto"`` or ``"mps"``)
        3. CPU (fallback)

    Args:
        priority: Device preference. One of ``"auto"``, ``"cuda"``, ``"mps"``,
            ``"cpu"``. Default is ``"auto"``.

    Returns:
        The selected ``torch.device``.

    Example:
        >>> device = get_device()  # Usually cuda:0
        >>> tensor = torch.randn(3, 3, device=device)
    """
    if priority in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    elif priority in ("auto", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    return device


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
    scheduler: Optional[Any] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint with full state for resumption.

    The checkpoint dictionary includes model state, optimizer state, current
    epoch, tracked metrics, optional LR scheduler state, and any additional
    user-provided data.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer (state dict extracted). May be ``None``.
        epoch: Current training epoch (1-based).
        metrics: Dictionary of scalar metrics, e.g.
            ``{"accuracy": 0.95, "eer": 0.02}``.
        path: Destination file path. Parent directories are created if needed.
        scheduler: Optional learning-rate scheduler to save.
        extra_data: Optional dictionary merged into the checkpoint.

    Raises:
        OSError: If the checkpoint cannot be written.

    Example:
        >>> save_checkpoint(model, optimizer, epoch=10,
        ...                 metrics={"eer": 0.015}, path="ckpt.pth")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    underlying = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": underlying.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra_data is not None:
        checkpoint.update(extra_data)

    torch.save(checkpoint, path)
    logging.info("Checkpoint saved to %s (epoch %d)", path, epoch)


def load_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint, optionally resuming optimizer / scheduler.

    Args:
        model: Model instance to load weights into.
        path: Checkpoint file path.
        optimizer: If provided, restores optimizer state (resumes training).
        scheduler: If provided, restores LR scheduler state.
        strict: Passed to ``model.load_state_dict``. Default ``True``.
        map_location: Device mapping for loading (e.g. ``"cpu"``).  ``None``
            lets PyTorch choose automatically.

    Returns:
        Dictionary containing checkpoint metadata: ``epoch``, ``metrics``,
        and any extra data keys.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If state dicts are incompatible.

    Example:
        >>> info = load_checkpoint(model, "ckpt.pth", optimizer=opt)
        >>> start_epoch = info["epoch"] + 1
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
        map_location = get_device().type

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Model weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Handle DataParallel / DistributedDataParallel prefixes
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=strict)

    # Optimizer
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Scheduler
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.info("Checkpoint loaded from %s (epoch %d)", path, checkpoint.get("epoch", "unknown"))
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        **{k: v for k, v in checkpoint.items() if k not in (
            "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch", "metrics"
        )},
    }


# ---------------------------------------------------------------------------
# Model analysis
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a PyTorch model.

    Args:
        model: PyTorch model.
        trainable_only: If ``True``, count only parameters that require grad.

    Returns:
        Number of (trainable) parameters.

    Example:
        >>> count_parameters(model)
        14200000
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> float:
    """Estimate FLOPs using a simple forward-pass hook.

    This is a lightweight estimation that counts multiply-add operations in
    ``nn.Conv2d``, ``nn.Linear``, and ``nn.MultiheadAttention`` layers.  It
    does not require external packages such as ``fvcore`` or ``ptflops``.

    Args:
        model: PyTorch model.
        input_size: Dummy input tensor shape (batch, *features).
        device: Device to run the dummy forward on. ``None`` uses CPU.

    Returns:
        Estimated FLOPs as a float (in GMACs, i.e. billions of operations).

    Note:
        This is a *simplified* estimate.  For precise FLOP counting use
        ``fvcore.nn.FlopCountAnalysis`` or ``ptflops``.
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device).eval()
    total_flops = 0
    hooks = []

    def _conv_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
        nonlocal total_flops
        batch_size = output.shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        # MACs per output position = kernel_ops * in_channels / groups
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
            module.in_channels // module.groups
        )
        output_size = batch_size * out_h * out_w * module.out_channels
        total_flops += output_size * kernel_ops

    def _linear_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
        nonlocal total_flops
        # MACs = in_features * out_features * batch_size
        total_flops += module.in_features * module.out_features * output.shape[0]

    def _attn_hook(module: nn.MultiheadAttention, input: Any, output: Any) -> None:
        nonlocal total_flops
        # Q, K, V projections + softmax(QK^T)V + output projection
        # Simplified: 4 * seq_len^2 * embed_dim * batch (for self-attn)
        # For cross-attn we approximate with average of seq lengths
        if isinstance(input, tuple) and len(input) >= 3:
            q, k, v = input[0], input[1], input[2]
            batch_size = q.shape[1]  # (seq, batch, embed)
            seq_len_q = q.shape[0]
            seq_len_k = k.shape[0]
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            # Q,K,V linear projections: 3 * seq * batch * embed^2
            proj_flops = 3 * (seq_len_q + seq_len_k + seq_len_k) * batch_size * embed_dim * embed_dim
            # Attention scores: batch * heads * seq_q * seq_k * head_dim
            head_dim = embed_dim // num_heads
            attn_flops = batch_size * num_heads * seq_len_q * seq_len_k * head_dim
            # Output projection
            out_proj_flops = seq_len_q * batch_size * embed_dim * embed_dim
            total_flops += proj_flops + 2 * attn_flops + out_proj_flops

    # Register hooks
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(_attn_hook))

    try:
        with torch.no_grad():
            dummy = torch.randn(input_size, device=device)
            if isinstance(dummy, (tuple, list)):
                model(*dummy)
            else:
                # Handle bimodal input (face + fingerprint)
                if input_size[0] == 2 and len(input_size) == 4:
                    model(dummy[0:1], dummy[1:2])
                else:
                    model(dummy)
    finally:
        for h in hooks:
            h.remove()

    gmacs = total_flops / 1e9
    logging.info("Estimated FLOPs: %.3f GMACs", gmacs)
    return gmacs


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


class Timer:
    """Context manager for timing code blocks.

    Records elapsed wall-clock time and optional GPU synchronization.

    Attributes:
        elapsed (float): Elapsed time in seconds (available after exit).

    Example:
        >>> with Timer("Forward pass") as t:
        ...     output = model(input)
        >>> print(f"Took {t.elapsed:.4f}s")
    """

    def __init__(self, name: str = "Block", sync_cuda: bool = True):
        """Initialise the timer.

        Args:
            name: Descriptive name printed in the log.
            sync_cuda: If ``True``, call ``torch.cuda.synchronize()`` before
                and after the timed block (more accurate GPU timing).
        """
        self.name = name
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        if self.sync_cuda:
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer and log elapsed time."""
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
        logging.info("[Timer] %s: %.4f s", self.name, self.elapsed)


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------


class AverageMeter:
    """Track running average of a scalar metric.

    Useful for loss and accuracy tracking during training loops.

    Attributes:
        val (float): Current (most recent) value.
        avg (float): Running average since last reset.
        sum (float): Accumulated sum.
        count (int): Number of accumulated values.

    Example:
        >>> meter = AverageMeter("loss")
        >>> for batch in loader:
        ...     meter.update(loss.item())
        >>> print(meter.avg)
    """

    def __init__(self, name: str = "metric") -> None:
        """Initialise the meter.

        Args:
            name: Human-readable name for logging.
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add a new value (or batch of values) to the accumulator.

        Args:
            val: Scalar value to accumulate.
            n: Number of samples represented by ``val`` (for batch averaging).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


class Logger:
    """Simple logging utility that writes simultaneously to console and file.

    Configures the root logger with a consistent format and optional file output.

    Example:
        >>> Logger.setup("logs/experiment.log", level=logging.INFO)
        >>> logging.info("Training started")
    """

    _configured: bool = False

    @classmethod
    def setup(
        cls,
        log_file: Optional[Union[str, Path]] = None,
        level: int = logging.INFO,
        fmt: Optional[str] = None,
    ) -> None:
        """Configure root logger for file + console output.

        Args:
            log_file: Path to log file. If ``None``, only console logging.
            level: Logging level (default ``logging.INFO``).
            fmt: Custom format string.  Default includes timestamp, level, msg.
        """
        if cls._configured:
            return

        if fmt is None:
            fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt=date_fmt)

        root = logging.getLogger()
        root.setLevel(level)
        root.handlers = []  # Clear existing handlers

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

        # File handler
        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

        cls._configured = True

    @classmethod
    def reset(cls) -> None:
        """Reset the configured flag (useful in tests)."""
        cls._configured = False


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def print_system_info() -> Dict[str, str]:
    """Print and return system configuration details.

    Returns:
        Dictionary with keys ``python``, ``pytorch``, ``cuda``, ``gpu``,
        ``device``.
    """
    info: Dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "N/A",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "device": str(get_device()),
    }

    header = "=" * 60
    logging.info(header)
    logging.info("UFM-Transformer System Information")
    logging.info(header)
    for key, value in info.items():
        logging.info("  %-12s: %s", key.upper(), value)
    logging.info(header)

    return info


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Self-test when module is run directly
    Logger.setup(level=logging.DEBUG)
    print_system_info()

    set_seed(42)

    device = get_device()
    print(f"Selected device: {device}")

    # Test AverageMeter
    meter = AverageMeter("test_loss")
    for i in range(5):
        meter.update(float(i), n=2)
    print(f"AverageMeter test: {meter}")
    assert abs(meter.avg - 2.0) < 1e-6, "AverageMeter calculation error"

    # Test Timer
    with Timer("sleep_test") as t:
        time.sleep(0.05)
    assert t.elapsed >= 0.05, "Timer did not capture elapsed time"

    # Test count_parameters
    dummy_model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )
    n_params = count_parameters(dummy_model)
    expected = 100 * 200 + 200 + 200 * 10 + 10  # weights + biases
    assert n_params == expected, f"Parameter count mismatch: {n_params} vs {expected}"
    print(f"Parameter count test passed: {n_params}")

    # Test checkpoint round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pth"
        save_checkpoint(
            dummy_model,
            None,
            epoch=5,
            metrics={"eer": 0.01},
            path=ckpt_path,
        )
        loaded = load_checkpoint(dummy_model, ckpt_path)
        assert loaded["epoch"] == 5
        assert loaded["metrics"]["eer"] == 0.01
        print("Checkpoint round-trip test passed")

    # Test compute_flops
    simple_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    flops = compute_flops(simple_conv, input_size=(1, 3, 224, 224), device=device)
    print(f"FLOPs estimate: {flops:.4f} GMACs")

    print("All utility tests passed.")
