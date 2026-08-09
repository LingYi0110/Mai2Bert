from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data.batching import DynamicBatchSampler, collate_events
from data.dataset import RegressionDataset
from lib.config.schema import AppConfig
from models.encoder import ChartEncoder
from preprocessing.representation import (
    CONTINUOUS_FIELDS,
    VOCAB_SIZES,
    schema_hash,
)
from preprocessing.store import ProcessedStore

MODEL_IMPLEMENTATION_VERSION = 4


def _group_tensorboard_status_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Put generic training status values under one TensorBoard namespace."""
    grouped: dict[str, float] = {}
    for name, value in metrics.items():
        if name == "epoch":
            grouped["status/epoch"] = value
        elif name == "lr-AdamW":
            grouped["status/lr"] = value
        elif name.startswith("lr-AdamW/"):
            grouped[f"status/lr/{name.removeprefix('lr-AdamW/')}"] = value
        else:
            grouped[name] = value
    return grouped


class GroupedTensorBoardLogger(TensorBoardLogger):
    """TensorBoard logger with a compact status namespace."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        super().log_metrics(_group_tensorboard_status_metrics(metrics), step=step)


def _cosine_with_warmup(
    optimizer: Optimizer,
    *,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Linear warmup → cosine decay to min_lr_ratio * base_lr."""

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step >= num_training_steps:
            return min_lr_ratio
        decay_steps = max(1, num_training_steps - warmup_steps)
        progress = float(current_step - warmup_steps) / float(decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return scale

    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: Optimizer,
    *,
    scheduler_name: str,
    num_training_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> LambdaLR | None:
    if scheduler_name == "none" or num_training_steps <= 0:
        return None
    warmup_steps = int(round(num_training_steps * warmup_ratio))
    return _cosine_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )


def resolve_dataset_binary_dir(config: AppConfig) -> Path:
    output = config.dataset_binary_dir
    if not output.is_dir():
        raise FileNotFoundError(
            f"processed dataset does not exist for experiment {config.experiment!r}: {output}"
        )
    return output


def set_encoder_trainability(
    encoder: ChartEncoder,
    *,
    trainable_layers: int | None,
    enabled: bool,
) -> None:
    """Freeze the encoder or expose only its last N Transformer blocks."""
    layers = list(encoder.transformer.children())
    if trainable_layers is not None and trainable_layers > len(layers):
        raise ValueError(
            f"trainable_encoder_layers={trainable_layers} exceeds encoder depth {len(layers)}"
        )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if not enabled:
        return
    if trainable_layers is None:
        for parameter in encoder.parameters():
            parameter.requires_grad = True
        return
    for layer in layers[-trainable_layers:] if trainable_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if trainable_layers:
        for module in (encoder.final_norm, encoder.pool_norm):
            for parameter in module.parameters():
                parameter.requires_grad = True


def build_encoder(config: AppConfig) -> ChartEncoder:
    return ChartEncoder(
        vocab_sizes=VOCAB_SIZES,
        continuous_fields=len(CONTINUOUS_FIELDS),
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        feedforward_size=config.model.feedforward_size,
        dropout=config.model.dropout,
        activation=config.model.activation,
        max_events=config.representation.max_events,
        stochastic_depth=config.model.stochastic_depth,
    )


def architecture_hash(config: AppConfig) -> str:
    payload = {
        "implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "schema_hash": schema_hash(),
        "model": config.model.model_dump(mode="json"),
        "max_events": config.representation.max_events,
        "vocab_sizes": VOCAB_SIZES,
        "continuous_fields": CONTINUOUS_FIELDS,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def regression_loader(
    store: ProcessedStore,
    config: AppConfig,
    split: Literal["train", "validation", "test"],
) -> tuple[RegressionDataset, DataLoader[dict[str, Any]]]:
    dataset = RegressionDataset(
        store,
        split,
        label_type=config.data.supervised_label_type,
        coarse_label_ranges=config.data.coarse_label_ranges,
        max_events=config.representation.max_events,
        stride=config.representation.stride,
    )
    sampler = DynamicBatchSampler(
        dataset.lengths,
        max_tokens=config.batch.max_tokens,
        max_batch_size=config.batch.max_size,
        bucket_size=config.batch.bucket_size,
        shuffle=split == "train",
        seed=config.trainer.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        collate_fn=partial(collate_events, pad_to_multiple=config.batch.pad_to_multiple),
    )
    return dataset, loader


class NamedModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` with a unique callback ``state_key``.

    Lightning requires unique state keys per callback type, which is what lets
    one trainer hold several checkpoints (best, periodic, latest) at once.

    With ``keep_latest_only`` the filename must contain an epoch placeholder
    (e.g. ``best-{epoch:02d}``): after each save, older files matching the
    same prefix are removed.
    """

    def __init__(
        self,
        state_key: str,
        *,
        keep_latest_only: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._state_key = state_key
        self._keep_latest_only = keep_latest_only

    @property
    def state_key(self) -> str:
        return self._state_key

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if not self._keep_latest_only:
            return
        saved = Path(filepath)
        match = re.match(r"^(?P<prefix>.*)-\d+(?P<suffix>\.ckpt)$", saved.name)
        if match is None:
            return
        for old in saved.parent.glob(f"{match.group('prefix')}-*{match.group('suffix')}"):
            if old != saved:
                old.unlink()


def resolve_resume_checkpoint(
    resume: str | Path | None,
    experiment_dir: Path,
) -> Path | None:
    """Resolve ``--resume``: an explicit file, or the newest ``last-*.ckpt``.

    Accepts a checkpoint file, ``"last"``, or an experiment directory; the
    latter two search ``experiment_dir``.
    """
    if resume is None:
        return None
    path = Path(resume)
    if path.is_file():
        return path
    search_dir = path if path.is_dir() else experiment_dir
    candidates = sorted(
        search_dir.glob("last-*.ckpt"),
        key=lambda p: int(re.search(r"-(\d+)\.ckpt$", p.name).group(1)),
    )
    if not candidates:
        raise FileNotFoundError(f"no last-*.ckpt checkpoint found to resume from ({resume})")
    newest = candidates[-1]
    logging.getLogger(__name__).info("resuming from %s", newest)
    return newest


def build_trainer(
    config: AppConfig,
    *,
    callbacks: list[Any] | None = None,
) -> L.Trainer:
    # Logs live directly in the experiment dir for tensorboard --logdir.
    experiment_dir = config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if config.trainer.limit_train_batches is not None:
        kwargs["limit_train_batches"] = config.trainer.limit_train_batches
    if config.trainer.limit_val_batches is not None:
        kwargs["limit_val_batches"] = config.trainer.limit_val_batches
    return L.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=config.trainer.deterministic,
        callbacks=callbacks,
        logger=[
            CSVLogger(save_dir=str(experiment_dir), name="", version=""),
            GroupedTensorBoardLogger(save_dir=str(experiment_dir), name="", version=""),
        ],
        default_root_dir=experiment_dir,
        **kwargs,
    )
