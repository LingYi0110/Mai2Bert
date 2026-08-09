from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .representation import EventArrays, representation_schema, schema_hash

STORE_FORMAT_VERSION = 6
_ARRAY_DATASETS = (
    "categorical",
    "categorical_presence",
    "continuous",
    "continuous_presence",
    "note_start",
)
_REQUIRED_DATASETS = (*_ARRAY_DATASETS, "offsets", "lengths")


def _schema_fields(schema: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = schema.get(key)
    if not isinstance(raw, (list, tuple)) or not all(isinstance(field, str) for field in raw):
        raise ValueError(f"schema {key!r} must be a sequence of strings")
    fields = tuple(raw)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError(f"schema {key!r} must be nonempty and contain unique names")
    return fields


def _attribute_text(value: Any, label: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(f"HDF5 attribute {label!r} must be text")


def _attribute_fields(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (bytes, str)):
        text = _attribute_text(value, label)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"HDF5 attribute {label!r} is not valid JSON") from error
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError(f"HDF5 attribute {label!r} must encode a string list")
        return tuple(decoded)
    if isinstance(value, np.ndarray):
        return tuple(_attribute_text(item, label) for item in value.tolist())
    raise ValueError(f"HDF5 attribute {label!r} has an unsupported type")


class ProcessedStoreWriter:
    """Append event arrays to a new immutable version 2 HDF5 store."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = dict(representation_schema() if schema is None else schema)
        self.categorical_fields = _schema_fields(self.schema, "categorical_fields")
        self.continuous_fields = _schema_fields(self.schema, "continuous_fields")
        self.file = h5py.File(self.path, "w")
        self.file.attrs["store_format_version"] = STORE_FORMAT_VERSION
        self.file.attrs["schema_hash"] = schema_hash(self.schema)
        self.file.attrs["categorical_fields"] = json.dumps(self.categorical_fields)
        self.file.attrs["continuous_fields"] = json.dumps(self.continuous_fields)

        categorical_count = len(self.categorical_fields)
        continuous_count = len(self.continuous_fields)
        self.categorical = self._matrix_dataset("categorical", categorical_count, "i2")
        self.categorical_presence = self._matrix_dataset(
            "categorical_presence", categorical_count, "bool"
        )
        self.continuous = self._matrix_dataset("continuous", continuous_count, "f4")
        self.continuous_presence = self._matrix_dataset(
            "continuous_presence", continuous_count, "bool"
        )
        self.note_start = self.file.create_dataset(
            "note_start",
            shape=(0,),
            maxshape=(None,),
            dtype="bool",
            chunks=True,
            compression="gzip",
        )
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self._offset = 0
        self._closed = False

    def _matrix_dataset(self, name: str, fields: int, dtype: str) -> h5py.Dataset:
        return self.file.create_dataset(
            name,
            shape=(0, fields),
            maxshape=(None, fields),
            dtype=dtype,
            chunks=True,
            compression="gzip",
        )

    def append(self, arrays: EventArrays) -> int:
        if self._closed:
            raise RuntimeError("cannot append to a closed processed store")
        expected_categorical = len(self.categorical_fields)
        expected_continuous = len(self.continuous_fields)
        if arrays.categorical.shape[1] != expected_categorical:
            raise ValueError(
                "categorical field dimension mismatch: "
                f"got {arrays.categorical.shape[1]}, expected {expected_categorical}"
            )
        if arrays.continuous.shape[1] != expected_continuous:
            raise ValueError(
                "continuous field dimension mismatch: "
                f"got {arrays.continuous.shape[1]}, expected {expected_continuous}"
            )

        row = len(self.offsets)
        end = self._offset + arrays.length
        for dataset, values in (
            (self.categorical, arrays.categorical),
            (self.categorical_presence, arrays.categorical_presence),
            (self.continuous, arrays.continuous),
            (self.continuous_presence, arrays.continuous_presence),
        ):
            dataset.resize((end, dataset.shape[1]))
            dataset[self._offset : end] = values
        self.note_start.resize((end,))
        self.note_start[self._offset : end] = arrays.note_start
        self.offsets.append(self._offset)
        self.lengths.append(arrays.length)
        self._offset = end
        return row

    def close(self) -> None:
        if self._closed:
            return
        self.file.create_dataset("offsets", data=np.asarray(self.offsets, dtype=np.int64))
        self.file.create_dataset("lengths", data=np.asarray(self.lengths, dtype=np.int32))
        self.file.flush()
        self.file.close()
        self._closed = True

    def __enter__(self) -> ProcessedStoreWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class StoredVariant:
    row: int
    music_id: str
    rotation: str
    split: str | None
    pretraining: bool
    difficulty_const: float | None
    length: int
    label_type: str | None = None
    dataset: str = "unknown"
    format: str = "json"
    physical_notes: int = 0
    slide_segments: int = 0
    parser_warnings: int = 0


class ProcessedStore:
    """Read a store only after validating its complete structural contract."""

    def __init__(
        self,
        directory: str | Path,
        *,
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "events.h5"
        if not self.path.is_file():
            raise FileNotFoundError(f"processed store does not exist: {self.path}")
        self.schema = dict(representation_schema() if schema is None else schema)
        self.categorical_fields = _schema_fields(self.schema, "categorical_fields")
        self.continuous_fields = _schema_fields(self.schema, "continuous_fields")
        self._file: h5py.File | None = None
        self.variants = self._load_variants(self.directory / "variants.jsonl")
        self._validate()

    @staticmethod
    def _load_variants(path: Path) -> list[StoredVariant]:
        records: list[StoredVariant] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    records.append(StoredVariant(**value))
                except (json.JSONDecodeError, TypeError) as error:
                    raise ValueError(
                        f"invalid variant manifest record at line {line_number}"
                    ) from error
        return records

    @staticmethod
    def _require_attributes(file: h5py.File, names: Sequence[str]) -> None:
        missing = [name for name in names if name not in file.attrs]
        if missing:
            raise ValueError(f"processed store is missing attributes: {', '.join(missing)}")

    def _validate(self) -> None:
        with h5py.File(self.path, "r") as file:
            self._require_attributes(
                file,
                (
                    "store_format_version",
                    "schema_hash",
                    "categorical_fields",
                    "continuous_fields",
                ),
            )
            version = int(file.attrs["store_format_version"])
            if version != STORE_FORMAT_VERSION:
                raise ValueError(
                    f"store format version mismatch: got {version}, expected {STORE_FORMAT_VERSION}"
                )
            actual_hash = _attribute_text(file.attrs["schema_hash"], "schema_hash")
            expected_hash = schema_hash(self.schema)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"schema hash mismatch: got {actual_hash!r}, expected {expected_hash!r}"
                )
            stored_categorical = _attribute_fields(
                file.attrs["categorical_fields"], "categorical_fields"
            )
            stored_continuous = _attribute_fields(
                file.attrs["continuous_fields"], "continuous_fields"
            )
            if stored_categorical != self.categorical_fields:
                raise ValueError("categorical_fields attribute does not match the expected schema")
            if stored_continuous != self.continuous_fields:
                raise ValueError("continuous_fields attribute does not match the expected schema")

            missing = [name for name in _REQUIRED_DATASETS if name not in file]
            if missing:
                raise ValueError(f"processed store is missing datasets: {', '.join(missing)}")
            categorical_rows = self._validate_matrix(
                file["categorical"],
                "categorical",
                len(self.categorical_fields),
                np.dtype(np.int16),
            )
            rows = self._validate_matrix(
                file["categorical_presence"],
                "categorical_presence",
                len(self.categorical_fields),
                np.dtype(np.bool_),
            )
            if rows != categorical_rows:
                raise ValueError("dataset 'categorical_presence' has an inconsistent row count")
            rows = self._validate_matrix(
                file["continuous"],
                "continuous",
                len(self.continuous_fields),
                np.dtype(np.float32),
            )
            if rows != categorical_rows:
                raise ValueError("dataset 'continuous' has an inconsistent row count")
            rows = self._validate_matrix(
                file["continuous_presence"],
                "continuous_presence",
                len(self.continuous_fields),
                np.dtype(np.bool_),
            )
            if rows != categorical_rows:
                raise ValueError("dataset 'continuous_presence' has an inconsistent row count")
            note_start = file["note_start"]
            if note_start.ndim != 1 or note_start.shape[0] != categorical_rows:
                raise ValueError("dataset 'note_start' must have one value per event row")
            if note_start.dtype != np.dtype(np.bool_):
                raise ValueError("dataset 'note_start' has an invalid dtype")

            offsets_dataset = file["offsets"]
            lengths_dataset = file["lengths"]
            if offsets_dataset.dtype.kind not in "iu" or lengths_dataset.dtype.kind not in "iu":
                raise ValueError("offsets and lengths must have integer dtypes")
            offsets = np.asarray(offsets_dataset)
            lengths = np.asarray(lengths_dataset)
            if offsets.ndim != 1 or lengths.ndim != 1 or offsets.shape != lengths.shape:
                raise ValueError(
                    "offsets and lengths must be equal-length one-dimensional datasets"
                )
            previous_end = 0
            for row, (offset_value, length_value) in enumerate(zip(offsets, lengths, strict=True)):
                offset = int(offset_value)
                length = int(length_value)
                if offset < 0 or length < 0 or offset + length > categorical_rows:
                    raise ValueError(f"offset range for row {row} is outside event datasets")
                if offset != previous_end:
                    raise ValueError(f"offset range for row {row} is not contiguous")
                previous_end = offset + length
            if previous_end != categorical_rows:
                raise ValueError("offset ranges do not cover all event rows")

            for variant in self.variants:
                if not 0 <= variant.row < len(offsets):
                    raise ValueError(f"variant row {variant.row} is outside the offsets dataset")
                if variant.length != int(lengths[variant.row]):
                    raise ValueError(
                        f"variant row {variant.row} length does not match the HDF5 store"
                    )

    @staticmethod
    def _validate_matrix(
        dataset: h5py.Dataset,
        name: str,
        fields: int,
        dtype: np.dtype[Any],
    ) -> int:
        if dataset.ndim != 2 or dataset.shape[1] != fields:
            raise ValueError(f"dataset {name!r} has an invalid field dimension")
        if dataset.dtype != dtype:
            raise ValueError(f"dataset {name!r} has an invalid dtype")
        return int(dataset.shape[0])

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def read(self, row: int) -> dict[str, np.ndarray[Any, Any]]:
        file = self._open()
        offsets = file["offsets"]
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(offsets):
            raise IndexError(f"processed store row out of range: {row!r}")
        offset = int(offsets[row])
        length = int(file["lengths"][row])
        selection = slice(offset, offset + length)
        return {name: np.asarray(file[name][selection]) for name in _ARRAY_DATASETS}

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> ProcessedStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        self.close()
