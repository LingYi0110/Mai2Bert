from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Subset

from data.batching import DynamicBatchSampler
from data.dataset import PretrainingDataset
from data.masking import MaskedEventBatchCollator, MaskedEventCollator
from lib.config.io import save_config
from lib.config.schema import AppConfig
from models.pretraining import MaskedEventModel
from preprocessing.representation import (
    CONTINUOUS_FIELDS,
    VOCAB_SIZES,
    schema_hash,
)
from preprocessing.store import ProcessedStore
from training.common import (
    NamedModelCheckpoint,
    architecture_hash,
    build_encoder,
    build_scheduler,
    build_trainer,
    resolve_dataset_binary_dir,
    resolve_resume_checkpoint,
)
from training.optimizers import build_optimizer


class PretrainingModule(L.LightningModule):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        encoder = build_encoder(config)
        self.model = MaskedEventModel(
            encoder,
            hidden_size=config.model.hidden_size,
            vocab_sizes=VOCAB_SIZES,
            continuous_fields=len(CONTINUOUS_FIELDS),
            objective_weights={
                "core": config.pretraining_loss.core_weight,
                "geometry": config.pretraining_loss.geometry_weight,
                "timing": config.pretraining_loss.timing_weight,
                "modifier": config.pretraining_loss.modifier_weight,
            },
        )
        self.save_hyperparameters(
            {
                "schema_hash": schema_hash(),
                "architecture_hash": architecture_hash(config),
                "config": config.model_dump(mode="json"),
            }
        )

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        losses = self.model.loss(self.model(batch), batch)
        target_count = batch["mask_target_count"]
        if not isinstance(target_count, int):
            raise TypeError("mask_target_count must remain a Python integer")
        for name, value in losses.items():
            if name.endswith("_count") or name == "loss":
                continue
            count = losses.get(f"{name}_count")
            # Field losses weight by their own valid target count; group losses are batch-averaged.
            batch_size = (
                1
                if name.endswith("_loss") and name != "token_loss"
                else (
                    max(int(count), 1) if isinstance(count, torch.Tensor) else max(target_count, 1)
                )
            )
            self.log(
                f"{stage}/{name}",
                value,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage == "val" and name == "objective_loss",
                batch_size=batch_size,
            )
        self.log(
            f"{stage}/target_count",
            float(target_count),
            on_step=stage == "train",
            on_epoch=True,
            batch_size=1,
        )
        return losses["loss"]

    def training_step(self, batch: dict[str, Any], _: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], _: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self) -> Any:
        optimizer = build_optimizer(
            self,
            self.config.optimizer,
            head_parameters=[
                *self.model.categorical_heads.parameters(),
                *self.model.continuous_head.parameters(),
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


def _group_rank(music_id: str) -> int:
    """Cheap 32-bit fingerprint for uniform sampling."""
    return zlib.crc32(music_id.encode("utf-8"))


def _pretraining_validation_groups(music_ids: list[str], ratio: float) -> set[str]:
    groups = sorted(set(music_ids))
    if not groups:
        return set()
    threshold = int(ratio * (1 << 32))
    selected = {music_id for music_id in groups if _group_rank(music_id) < threshold}
    if len(groups) > 1 and not selected:
        selected.add(min(groups, key=lambda music_id: (_group_rank(music_id), music_id)))
    if len(groups) > 1 and len(selected) == len(groups):
        selected.remove(max(groups, key=lambda music_id: (_group_rank(music_id), music_id)))
    return selected


def _pretraining_loaders(
    store: ProcessedStore, config: AppConfig
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_dataset = PretrainingDataset(
        store,
        max_events=config.representation.max_events,
        seed=config.trainer.seed,
    )
    validation_dataset = PretrainingDataset(
        store,
        max_events=config.representation.max_events,
        seed=config.trainer.seed,
    )
    canonical_groups = {variant.music_id for variant in train_dataset.variants}
    if len(canonical_groups) < 2:
        raise ValueError("pretraining requires at least two canonical groups")
    validation_groups = _pretraining_validation_groups(
        [variant.music_id for variant in train_dataset.variants],
        config.split.validation_ratio,
    )
    evaluation_rotations = set(config.augmentation.evaluation_rotations)
    validation_indices = [
        index
        for index, variant in enumerate(validation_dataset.variants)
        if variant.music_id in validation_groups and variant.rotation in evaluation_rotations
    ]
    train_indices = [
        index
        for index, variant in enumerate(train_dataset.variants)
        if variant.music_id not in validation_groups
    ]
    if not train_indices or not validation_indices:
        raise ValueError("pretraining split produced an empty train or validation subset")
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    def masking(*, deterministic_seed: int | None = None) -> MaskedEventBatchCollator:
        return MaskedEventBatchCollator(
            MaskedEventCollator(
                probability=config.masking.probability,
                mask_probability=config.masking.mask_probability,
                random_probability=config.masking.random_probability,
                vocab_sizes=VOCAB_SIZES,
                deterministic_seed=deterministic_seed,
            ),
            pad_to_multiple=config.batch.pad_to_multiple,
        )

    def loader(
        subset: Subset[Any], *, shuffle: bool, collate_fn: MaskedEventBatchCollator
    ) -> DataLoader[Any]:
        parent = subset.dataset
        if not isinstance(parent, PretrainingDataset):
            raise TypeError("pretraining subset must wrap PretrainingDataset")
        # Cache lengths: rebuilding per subset index was quadratic.
        lengths = [
            min(parent.variants[index].length, config.representation.max_events)
            for index in subset.indices
        ]
        sampler = DynamicBatchSampler(
            lengths,
            max_tokens=config.batch.max_tokens,
            max_batch_size=config.batch.max_size,
            bucket_size=config.batch.bucket_size,
            shuffle=shuffle,
            seed=config.trainer.seed,
        )
        return DataLoader(
            subset,
            batch_sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            collate_fn=collate_fn,
        )

    return loader(train_subset, shuffle=True, collate_fn=masking()), loader(
        validation_subset,
        shuffle=False,
        collate_fn=masking(deterministic_seed=config.trainer.seed),
    )


def run_pretrain(
    config: AppConfig,
    *,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    L.seed_everything(config.trainer.seed, workers=True)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config.model_dump(mode="json"), config.experiment_dir / "config.yaml")
    store = ProcessedStore(resolve_dataset_binary_dir(config))

    train_loader, validation_loader = _pretraining_loaders(store, config)
    checkpoint = NamedModelCheckpoint(
        state_key="best",
        keep_latest_only=True,
        dirpath=config.experiment_dir,
        monitor="val/objective_loss",
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
        filename="pretrain-epoch-{epoch:02d}",
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
            checkpoint,
            periodic_checkpoint,
            latest_checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(
        PretrainingModule(config),
        train_loader,
        validation_loader,
        ckpt_path=str(resolve_resume_checkpoint(resume_checkpoint, config.experiment_dir))
        if resume_checkpoint is not None
        else None,
    )
    store.close()
    return Path(checkpoint.best_model_path or latest_checkpoint.best_model_path)
