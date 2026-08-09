from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor

from data.dataset import RegressionDataset
from lib.config.io import save_config
from lib.config.schema import AppConfig
from models.encoder import ChartEncoder
from models.regression import DifficultyRegressor
from preprocessing.representation import schema_hash
from preprocessing.store import ProcessedStore
from training.common import (
    NamedModelCheckpoint,
    architecture_hash,
    build_encoder,
    build_scheduler,
    build_trainer,
    regression_loader,
    resolve_resume_checkpoint,
)
from training.ema import EmaMixin
from training.losses import (
    gaussian_interval_nll,
    gaussian_point_nll,
)
from training.metrics import flatten_metrics, regression_metrics
from training.optimizers import build_optimizer
from training.transfer import transfer_encoder


def set_encoder_trainability(
    encoder: ChartEncoder,
    *,
    trainable_layers: int | None,
    enabled: bool,
) -> None:
    """Freeze the encoder or expose only its last N Transformer blocks.

    Delegates to :func:`training.common.set_encoder_trainability`.
    """
    from training.common import set_encoder_trainability as _set_trainability

    _set_trainability(encoder, trainable_layers=trainable_layers, enabled=enabled)


class RegressionModule(EmaMixin):
    def __init__(
        self,
        config: AppConfig,
        *,
        target_mean: float,
        target_std: float,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = DifficultyRegressor(
            build_encoder(config),
            hidden_size=config.model.hidden_size,
            regression_hidden_size=config.model.regression_hidden_size,
            max_events=config.representation.max_events,
            stride=config.representation.stride,
            dropout=config.model.dropout,
            aggregator_heads=config.model.aggregator_heads or config.model.num_heads,
            aggregator_layers=config.model.aggregator_layers,
            init_sigma=config.loss.gaussian_init_sigma,
        )
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(max(target_std, 1e-6), dtype=torch.float32))
        self.validation_predictions: list[float] = []
        self.validation_targets: list[float] = []
        self.validation_groups: list[str] = []
        self.validation_sigmas: list[float] = []
        self.test_metrics: dict[str, Any] = {}
        self.save_hyperparameters(
            {
                "target_mean": target_mean,
                "target_std": target_std,
                "schema_hash": schema_hash(),
                "architecture_hash": architecture_hash(config),
                "config": config.model_dump(mode="json"),
            }
        )

    def predict_gaussian(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw-space predictions and transformed-label-space sigma."""
        mean, sigma = self.model.forward_gaussian(batch)
        transformed: torch.Tensor = mean * self.target_std + self.target_mean
        return _label_inverse_transform(self.config.data.label_transform)(transformed), sigma

    def predict_raw(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        standardized = self.model(batch)
        transformed: torch.Tensor = standardized * self.target_std + self.target_mean
        return _label_inverse_transform(self.config.data.label_transform)(transformed)

    def _gaussian_training_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        standardized_mean, standardized_sigma = self.model.forward_gaussian(batch)
        transformed = standardized_mean * self.target_std + self.target_mean
        sigma = standardized_sigma * self.target_std
        transform = _label_transform(self.config.data.label_transform)
        target_transformed = transform(batch["target"])
        precise = ~batch["is_coarse"]
        losses: list[torch.Tensor] = []
        loss_weights: list[torch.Tensor] = []
        if precise.any():
            losses.append(
                gaussian_point_nll(
                    transformed[precise],
                    sigma[precise],
                    target_transformed[precise],
                )
            )
            loss_weights.append(precise.sum().to(transformed.dtype))
        if batch["is_coarse"].any():
            coarse = batch["is_coarse"]
            losses.append(
                gaussian_interval_nll(
                    transformed[coarse],
                    sigma[coarse],
                    transform(batch["coarse_min"][coarse]),
                    transform(batch["coarse_max"][coarse]),
                )
            )
            loss_weights.append(coarse.sum().to(transformed.dtype))
        if not losses:
            raise ValueError("batch contains no supervised targets")
        total_weight = torch.stack(loss_weights).sum().clamp_min(1e-8)
        loss = (
            sum(value * weight for value, weight in zip(losses, loss_weights, strict=True))
            / total_weight
        )
        predictions = _label_inverse_transform(self.config.data.label_transform)(transformed)
        return loss, predictions, sigma

    def training_step(self, batch: dict[str, Any], _: int) -> torch.Tensor:
        loss, predictions, sigma = self._gaussian_training_loss(batch)
        high = batch["target"] >= self.config.loss.threshold
        batch_size = int(batch["target"].shape[0])
        coarse = batch["is_coarse"]
        if coarse.any():
            interval_errors = torch.maximum(
                batch["coarse_min"][coarse] - predictions[coarse],
                torch.maximum(
                    predictions[coarse] - batch["coarse_max"][coarse],
                    torch.zeros_like(predictions[coarse]),
                ),
            )
            self.log(
                "train/coarse_in_range_rate",
                (interval_errors <= 0).float().mean(),
                on_step=False,
                on_epoch=True,
                batch_size=int(coarse.sum()),
            )
            self.log(
                "train/coarse_boundary_mae",
                interval_errors.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=int(coarse.sum()),
            )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/sigma_mean",
            sigma.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if high.any():
            self.log(
                "train/high_mae",
                torch.mean(torch.abs(predictions[high] - batch["target"][high])),
                on_step=False,
                on_epoch=True,
                batch_size=int(high.sum()),
            )
        return loss

    def on_train_epoch_start(self) -> None:
        enabled = self.current_epoch >= self.config.trainer.freeze_encoder_epochs
        set_encoder_trainability(
            self.model.encoder,
            trainable_layers=self.config.trainer.trainable_encoder_layers,
            enabled=enabled,
        )

    def test_step(self, batch: dict[str, Any], _: int) -> None:
        # Same buffers and metrics path as validation.
        self.validation_step(batch, _)

    def on_test_epoch_end(self) -> None:
        if not self.validation_targets:
            self.test_metrics = {}
            return
        metrics = regression_metrics(
            self.validation_predictions,
            self.validation_targets,
            self.validation_groups,
            threshold=self.config.loss.threshold,
            bin_width=self.config.loss.bin_width,
        )
        output = self.config.experiment_dir
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.test_metrics = metrics

    def validation_step(self, batch: dict[str, Any], _: int) -> None:
        predictions, sigma = self.predict_gaussian(batch)
        self.validation_predictions.extend(predictions.detach().cpu().tolist())
        self.validation_targets.extend(batch["target"].detach().cpu().tolist())
        self.validation_groups.extend(batch["music_id"])
        self.validation_sigmas.extend(sigma.detach().cpu().tolist())

    def on_validation_epoch_end(self) -> None:
        if not self.validation_targets:
            return
        metrics = regression_metrics(
            self.validation_predictions,
            self.validation_targets,
            self.validation_groups,
            threshold=self.config.loss.threshold,
            bin_width=self.config.loss.bin_width,
        )
        flattened = flatten_metrics(metrics, "val")
        val_batch_size = len(self.validation_targets)
        for name, value in flattened.items():
            if math.isfinite(value):
                self.log(name, value, sync_dist=False, batch_size=val_batch_size)
        high_key = f"ge_{self.config.loss.threshold:g}".replace(".", "_")
        high_mae = float(metrics[high_key]["mae"])
        if not math.isfinite(high_mae):
            high_mae = float(metrics["mae"])
        self.log("val/high_mae", high_mae, prog_bar=True, batch_size=val_batch_size)
        sigmas = np.asarray(self.validation_sigmas, dtype=np.float64)
        targets_array = np.asarray(self.validation_targets, dtype=np.float64)
        errors = np.asarray(self.validation_predictions, dtype=np.float64) - targets_array
        high_mask = targets_array >= self.config.loss.threshold
        if len(sigmas) == len(errors):
            self.log("val/sigma_mean", float(sigmas.mean()), batch_size=val_batch_size)
            if high_mask.any():
                self.log(
                    "val/sigma_high_mean",
                    float(sigmas[high_mask].mean()),
                    batch_size=val_batch_size,
                )
            self.log(
                "val/sigma_rmse",
                float(np.sqrt(np.mean((errors**2 - sigmas**2) ** 2))),
                batch_size=val_batch_size,
            )
        self.validation_predictions.clear()
        self.validation_targets.clear()
        self.validation_groups.clear()
        self.validation_sigmas.clear()
        super().on_validation_epoch_end()

    def configure_optimizers(self) -> Any:
        optimizer = build_optimizer(
            self,
            self.config.optimizer,
            head_parameters=[
                *self.model.aggregator.parameters(),
                *self.model.head.parameters(),
                *self.model.uncertainty_head.parameters(),
            ],
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_name=self.config.optimizer.scheduler,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
            warmup_ratio=self.config.optimizer.warmup_ratio,
            min_lr_ratio=self.config.optimizer.min_lr_ratio,
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def _label_transform(mode: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the forward label transform for ``mode`` (identity/log/sqrt)."""
    if mode == "log":
        return torch.log
    if mode == "sqrt":
        return torch.sqrt
    if mode == "identity":
        return lambda value: value
    raise ValueError(f"unsupported label_transform: {mode}")


def _label_inverse_transform(mode: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the inverse label transform (raw difficulty space)."""
    if mode == "log":
        return torch.exp
    if mode == "sqrt":
        return lambda value: value * value
    if mode == "identity":
        return lambda value: value
    raise ValueError(f"unsupported label_transform: {mode}")


def _training_statistics(
    store: ProcessedStore,
    label_type: str = "all",
    *,
    coarse_label_ranges: dict[str, tuple[float, float | None]] | None = None,
    label_transform: str = "identity",
) -> tuple[float, float]:
    dataset = RegressionDataset(
        store,
        "train",
        label_type=label_type,
        coarse_label_ranges=coarse_label_ranges,
    )
    identity_targets = [
        float(variant.difficulty_const)
        for variant in dataset.variants
        if variant.rotation == "Identity" and variant.difficulty_const is not None
    ]
    if not identity_targets:
        raise ValueError("training split contains no identity labels")
    values = np.asarray(identity_targets, dtype=np.float64)
    if label_transform == "log":
        transformed = np.log(values)
    elif label_transform == "sqrt":
        transformed = np.sqrt(values)
    else:
        transformed = values
    # Mean/std are computed in the transformed space the model regresses.
    return float(transformed.mean()), float(transformed.std())


def run_finetune(
    config: AppConfig,
    *,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    """Train the difficulty regressor on top of the pretrained encoder."""
    L.seed_everything(config.trainer.seed, workers=True)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config.model_dump(mode="json"), config.experiment_dir / "config.yaml")
    store = ProcessedStore(config.dataset_binary_dir)
    target_mean, target_std = _training_statistics(
        store,
        label_type=config.data.supervised_label_type,
        coarse_label_ranges=config.data.coarse_label_ranges,
        label_transform=config.data.label_transform,
    )
    module = RegressionModule(config, target_mean=target_mean, target_std=target_std)
    if config.transfer.checkpoint is not None:
        transfer_encoder(
            module.model.encoder,
            config.transfer.checkpoint,
            expected_schema_hash=schema_hash(),
            expected_architecture_hash=architecture_hash(config),
            strict_architecture=config.transfer.strict_architecture,
        )
    _, train_loader = regression_loader(store, config, "train")
    _, validation_loader = regression_loader(store, config, "validation")
    best_checkpoint = NamedModelCheckpoint(
        state_key="best",
        keep_latest_only=True,
        dirpath=config.experiment_dir,
        monitor="val/mae",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}",
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    periodic_checkpoint = NamedModelCheckpoint(
        state_key="periodic",
        dirpath=config.experiment_dir,
        save_top_k=-1,
        every_n_epochs=config.trainer.checkpoint_every_n_epochs,
        filename="finetune-epoch-{epoch:02d}",
        auto_insert_metric_name=False,
    )
    latest_checkpoint = NamedModelCheckpoint(
        state_key="last",
        keep_latest_only=True,
        dirpath=config.experiment_dir,
        save_top_k=1,
        every_n_epochs=1,
        filename="last-{epoch:02d}",
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    trainer = build_trainer(
        config,
        callbacks=[
            best_checkpoint,
            periodic_checkpoint,
            latest_checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(
        module,
        train_loader,
        validation_loader,
        ckpt_path=str(resolve_resume_checkpoint(resume_checkpoint, config.experiment_dir))
        if resume_checkpoint is not None
        else None,
    )
    return Path(best_checkpoint.best_model_path or latest_checkpoint.best_model_path)


@torch.no_grad()
def run_evaluate(config: AppConfig) -> dict[str, Any]:
    """Evaluate the finetuned regressor on the held-out test split.

    Runs through ``trainer.test`` so the RegressionModule's test hooks own the
    metric computation and ``metrics.json`` write (same code path as the
    module's validation hooks).
    """
    from inference.loader import load_module

    checkpoint = config.paths.checkpoint or config.transfer.checkpoint
    if checkpoint is None:
        raise ValueError("evaluate requires paths.checkpoint")
    save_config(config.model_dump(mode="json"), config.experiment_dir / "config.yaml")
    module = load_module(config, checkpoint)
    store = ProcessedStore(config.dataset_binary_dir)
    _, loader = regression_loader(store, config, "test")
    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        precision=config.trainer.precision,
    )
    trainer.test(module, loader)
    return module.test_metrics


def run_predict(
    config: AppConfig,
    *,
    chart: str,
    format: str | None = None,
    difficulty: int | None = None,
) -> float:
    """Predict difficulty for one chart file (MA2 or Simai text)."""
    from inference.loader import load_module
    from inference.predict import predict_file

    module = load_module(config)
    result = predict_file(
        module,
        config,
        chart,
        format=format,
        difficulty=difficulty,
    )
    return float(result.predicted)
