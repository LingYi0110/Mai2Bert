from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Rotation = Literal[
    "Identity",
    "Clockwise90",
    "Clockwise180",
    "Counterclockwise90",
]


def _default_train_rotations() -> list[Rotation]:
    return ["Identity", "Clockwise90", "Clockwise180", "Counterclockwise90"]


def _default_evaluation_rotations() -> list[Rotation]:
    return ["Identity"]


def _default_datasets() -> list[str]:
    return ["official", "thirdparty"]


def _default_coarse_label_ranges() -> dict[str, tuple[float, float | None]]:
    # Observed official ranges ("13+" -> "13.5"); null = open upper bound.
    return {
        "10": (9.913, 10.6897),
        "10.5": (10.4027, 11.1829),
        "11": (10.8887, 11.7112),
        "11.5": (11.4151, 12.1507),
        "12": (11.9005, 12.7011),
        "12.5": (12.3664, 13.2002),
        "13": (12.8597, 13.6855),
        "13.5": (13.2946, 14.1787),
        "14": (13.7095, 14.681),
        "14.5": (14.1362, 15.1938),
        "15": (15.2113, 15.2762),
        "15.5": (15.0, None),
    }


def _coarse_label_key(value: float) -> str:
    return f"{float(value):g}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class PathsConfig(StrictModel):
    raw_data: Path = Path("datasets")
    binary_root: Path = Path("binary")
    experiments_root: Path = Path("experiments")
    # Processed dataset directory, shared by all training stages.
    dataset_experiment: Path = Path("v1")
    checkpoint: Path | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_paths(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value)
        return value


class DataConfig(StrictModel):
    supervised_label_type: Literal["all", "precise", "coarse"] = "all"
    # Target transform before standardization; predictions return in raw difficulty.
    label_transform: Literal["identity", "log", "sqrt"] = "identity"
    # Coarse predictions penalized only outside their label range.
    coarse_label_ranges: dict[str, tuple[float, float | None]] = Field(
        default_factory=_default_coarse_label_ranges,
        min_length=1,
    )
    # Separate worker pools: prepare parses charts, DataLoader serves HDF5 rows.
    num_workers: int = Field(0, ge=0)
    preprocessing_workers: int = Field(0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False

    @field_validator("coarse_label_ranges", mode="before")
    @classmethod
    def _coerce_coarse_label_ranges(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        coerced: dict[str, tuple[float, float | None]] = {}
        for raw_key, raw_range in value.items():
            key = str(raw_key)
            minimum: float
            maximum: float | None
            if isinstance(raw_range, Mapping):
                minimum = float(raw_range["min"])
                maximum_value = raw_range.get("max")
                maximum = None if maximum_value is None else float(maximum_value)
            else:
                sequence = list(raw_range) if not isinstance(raw_range, str) else [raw_range]
                if len(sequence) != 2:
                    raise ValueError(f"coarse label range {key!r} must contain exactly min and max")
                minimum = float(sequence[0])
                maximum = None if sequence[1] is None else float(sequence[1])
            if not math.isfinite(minimum):
                raise ValueError(f"coarse label range {key!r} minimum must be finite")
            if maximum is not None and not math.isfinite(maximum):
                raise ValueError(f"coarse label range {key!r} maximum must be finite or null")
            coerced[_coarse_label_key(float(key))] = (minimum, maximum)
        return coerced

    @model_validator(mode="after")
    def _workers_support_persistence(self) -> DataConfig:
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        return self

    def coarse_range(self, label_value: float) -> tuple[float, float | None] | None:
        return self.coarse_label_ranges.get(_coarse_label_key(label_value))


class SplitConfig(StrictModel):
    train_ratio: float = Field(0.8, gt=0, lt=1)
    validation_ratio: float = Field(0.1, gt=0, lt=1)
    test_ratio: float = Field(0.1, gt=0, lt=1)
    seed: int = 42
    high_diff_threshold: float = 12.0
    bin_width: float = Field(0.1, gt=0)
    rare_bin_max_groups: int = Field(2, ge=1)

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SplitConfig:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


class RepresentationConfig(StrictModel):
    max_events: int = Field(512, ge=8)
    stride: int = Field(384, ge=1)
    continuous_clip: float = Field(10.0, gt=0)

    @model_validator(mode="after")
    def _stride_fits_window(self) -> RepresentationConfig:
        if self.stride > self.max_events:
            raise ValueError("representation.stride cannot exceed max_events")
        return self


class AugmentationConfig(StrictModel):
    train_rotations: list[Rotation] = Field(default_factory=_default_train_rotations)
    evaluation_rotations: list[Rotation] = Field(default_factory=_default_evaluation_rotations)
    train_timing_jitter_seconds: float = Field(0.005, ge=0.0)

    @model_validator(mode="after")
    def _rotations_are_unique(self) -> AugmentationConfig:
        if len(set(self.train_rotations)) != len(self.train_rotations):
            raise ValueError("train_rotations must not contain duplicates")
        if len(set(self.evaluation_rotations)) != len(self.evaluation_rotations):
            raise ValueError("evaluation_rotations must not contain duplicates")
        return self


class ModelConfig(StrictModel):
    hidden_size: int = Field(256, ge=32)
    num_layers: int = Field(6, ge=1)
    num_heads: int = Field(8, ge=1)
    feedforward_size: int = Field(1024, ge=64)
    dropout: float = Field(0.1, ge=0, lt=1)
    activation: Literal["gelu", "relu"] = "gelu"
    regression_hidden_size: int = Field(256, ge=16)
    stochastic_depth: float = Field(0.0, ge=0, lt=1)
    aggregator_heads: int | None = None
    aggregator_layers: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _heads_divide_hidden_size(self) -> ModelConfig:
        if self.hidden_size % self.num_heads:
            raise ValueError("model.hidden_size must be divisible by num_heads")
        if self.aggregator_heads is not None and self.hidden_size % self.aggregator_heads:
            raise ValueError("model.aggregator_heads must divide model.hidden_size")
        return self


class BatchConfig(StrictModel):
    max_tokens: int = Field(8192, ge=8)
    max_size: int = Field(32, ge=1)
    bucket_size: int = Field(256, ge=1)
    pad_to_multiple: int = Field(8, ge=1)


class MaskingConfig(StrictModel):
    # Whole NOTE-compound masking (one row per compound).
    probability: float = Field(0.15, gt=0, lt=1)
    mask_probability: float = Field(0.8, ge=0, le=1)
    random_probability: float = Field(0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _replacement_probabilities_fit_one(self) -> MaskingConfig:
        total = self.mask_probability + self.random_probability
        if total > 1.0 + 1e-8:
            raise ValueError(f"mask and random probabilities must not exceed 1.0, got {total}")
        return self


class PretrainingLossConfig(StrictModel):
    """Relative weights for field-balanced masked-event reconstruction."""

    core_weight: float = Field(1.0, gt=0)
    geometry_weight: float = Field(2.0, gt=0)
    # Smooth-L1 z-scored values need extra weight vs cross-entropy.
    timing_weight: float = Field(3.0, gt=0)
    modifier_weight: float = Field(0.25, gt=0)


class LossConfig(StrictModel):
    # Gaussian NLL (point for precise labels, interval mass for coarse); also buckets eval metrics.
    threshold: float = 12.0
    bin_width: float = Field(0.1, gt=0)
    # Initial raw sigma; below 1 avoids over-confident NLL gradients.
    gaussian_init_sigma: float = Field(0.5, gt=0)


class OptimizerConfig(StrictModel):
    name: Literal["adamw", "muon_hybrid"] = "adamw"
    encoder_lr: float = Field(1e-4, gt=0)
    head_lr: float = Field(5e-4, gt=0)
    # match_rms_adamw scales the update so muon_lr tunes like an AdamW LR.
    muon_lr: float = Field(0.005, gt=0)
    muon_momentum: float = Field(0.95, gt=0, lt=1)
    muon_adjust_lr: Literal["original", "match_rms_adamw"] = "match_rms_adamw"
    adam_epsilon: float = Field(1e-8, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    beta1: float = Field(0.9, gt=0, lt=1)
    beta2: float = Field(0.999, gt=0, lt=1)
    scheduler: Literal["none", "cosine"] = "cosine"
    warmup_ratio: float = Field(0.05, ge=0, le=0.5)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)


class TrainerConfig(StrictModel):
    seed: int = 42
    max_epochs: int = Field(20, ge=1)
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: Literal[
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
    ] = "32-true"
    gradient_clip_val: float = Field(1.0, ge=0)
    accumulate_grad_batches: int = Field(1, ge=1)
    log_every_n_steps: int = Field(10, ge=1)
    # Archive a full training state every N epochs (*-epoch-<NN>.ckpt).
    checkpoint_every_n_epochs: int = Field(5, ge=1)
    freeze_encoder_epochs: int = Field(1, ge=0)
    # None unfreezes all; a number unfreezes only the last N Transformer blocks.
    trainable_encoder_layers: int | None = Field(None, ge=0)
    deterministic: bool = True
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None


class TransferConfig(StrictModel):
    checkpoint: Path | None = None
    strict_architecture: bool = True

    @field_validator("checkpoint", mode="before")
    @classmethod
    def _coerce_checkpoint(cls, value: object) -> Path | None:
        if isinstance(value, str):
            return Path(value) if value else None
        if value is None or isinstance(value, Path):
            return value
        return Path(str(value))


class LoggingConfig(StrictModel):
    level: str = "INFO"
    file: Path | None = None
    # Log directory; when file is null, logs are written to <log_dir>/mai2bert.log.
    log_dir: Path = Path("logs")
    rotation: str = "20 MB"
    retention: str = "14 days"

    @field_validator("file", mode="before")
    @classmethod
    def _coerce_file(cls, value: object) -> Path | None:
        if isinstance(value, str):
            return Path(value) if value else None
        if value is None or isinstance(value, Path):
            return value
        return Path(str(value))

    @field_validator("log_dir", mode="before")
    @classmethod
    def _coerce_log_dir(cls, value: object) -> Path:
        if isinstance(value, str):
            return Path(value)
        if isinstance(value, Path):
            return value
        return Path(str(value))


class AppConfig(StrictModel):
    experiment: str = Field("experiment1", min_length=1)
    datasets: list[str] = Field(default_factory=_default_datasets, min_length=1)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    representation: RepresentationConfig = Field(default_factory=RepresentationConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    pretraining_loss: PretrainingLossConfig = Field(default_factory=PretrainingLossConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _validate_cross_section_constraints(self) -> AppConfig:
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("dataset folders must be unique")
        if (
            not self.experiment
            or self.experiment != self.experiment.strip()
            or Path(self.experiment).name != self.experiment
            or self.experiment in {".", ".."}
        ):
            raise ValueError("experiment must be a single non-empty directory name")
        if (
            not self.paths.dataset_experiment
            or self.paths.dataset_experiment.as_posix() != self.paths.dataset_experiment.name
            or self.paths.dataset_experiment.name in {".", ".."}
        ):
            raise ValueError("paths.dataset_experiment must be a single non-empty directory name")
        for folder in self.datasets:
            if (
                not folder
                or folder != folder.strip()
                or Path(folder).name != folder
                or folder in {".", ".."}
            ):
                raise ValueError("dataset entries must be direct folder names")
        if self.batch.max_tokens < self.representation.max_events:
            raise ValueError("batch.max_tokens must fit at least one representation window")
        if not math.isclose(
            self.loss.threshold,
            self.split.high_diff_threshold,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("loss.threshold and split.high_diff_threshold must match")
        if not math.isclose(self.loss.bin_width, self.split.bin_width, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("loss.bin_width and split.bin_width must match")
        if (
            self.trainer.trainable_encoder_layers is not None
            and self.trainer.trainable_encoder_layers > self.model.num_layers
        ):
            raise ValueError("trainer.trainable_encoder_layers cannot exceed model.num_layers")
        return self

    @property
    def dataset_binary_dir(self) -> Path:
        return self.paths.binary_root / self.paths.dataset_experiment

    @property
    def experiment_dir(self) -> Path:
        return self.paths.experiments_root / self.experiment
