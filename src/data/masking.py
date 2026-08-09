from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from data.batching import collate_events
from preprocessing import representation

SPECIAL_TOKENS = representation.SPECIAL_TOKENS


@dataclass(frozen=True, slots=True)
class _Compound:
    """Compact coordinates for a physical NOTE and its SLIDE_SEGMENT rows."""

    row: int
    columns: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.columns.numel())


@dataclass(slots=True)
class MaskedEventCollator:
    probability: float = 0.15
    mask_probability: float = 0.8
    random_probability: float = 0.1
    vocab_sizes: tuple[int, ...] = ()
    deterministic_seed: int | None = None

    def _generator(self, batch: dict[str, Any], device: torch.device) -> torch.Generator | None:
        if self.deterministic_seed is None:
            return None
        digest = hashlib.sha256(str(self.deterministic_seed).encode())
        for key in ("dataset", "music_id", "rotation"):
            digest.update(key.encode())
            digest.update(repr(batch.get(key)).encode())
        seed = int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)
        return torch.Generator(device=device.type).manual_seed(seed)

    @staticmethod
    def _compounds(
        valid_events: torch.Tensor,
        note_start: torch.Tensor,
    ) -> list[_Compound]:
        """Return compact coordinates for every NOTE compound in the batch.

        Do not construct a full ``[batch, sequence]`` boolean mask per note:
        large batches contain thousands of notes, and doing so made collation
        the bottleneck that periodically starved the GPU.
        """
        compounds: list[_Compound] = []
        for row in range(valid_events.shape[0]):
            valid_indices = torch.nonzero(valid_events[row], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            starts = torch.nonzero(note_start[row, valid_indices], as_tuple=False).flatten()
            # Charts start with a NOTE; keep tolerant fallback for a malformed first flag.
            if starts.numel() == 0 or int(starts[0]) != 0:
                starts = torch.cat((starts.new_zeros(1), starts))
            boundaries = [*starts.tolist(), int(valid_indices.numel())]
            for start, end in zip(boundaries, boundaries[1:], strict=False):
                compounds.append(_Compound(row=row, columns=valid_indices[start:end]))
        return compounds

    @staticmethod
    def _select_compounds(
        compounds: list[_Compound],
        probability: float,
        *,
        shape: torch.Size,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, list[int]]:
        selected = torch.zeros(shape, dtype=torch.bool, device=device)
        # Selection is uniform over compounds (one NOTE row per compound).
        probabilities = torch.full((len(compounds),), probability, device=device)
        selected_indices = (
            torch.nonzero(
                torch.rand(len(compounds), device=device, generator=generator) < probabilities,
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        for index in selected_indices:
            compound = compounds[index]
            selected[compound.row, compound.columns] = True
        return selected, selected_indices

    @staticmethod
    def _compound_actions(
        compounds: list[_Compound],
        selected_indices: list[int],
        *,
        shape: torch.Size,
        mask_probability: float,
        random_probability: float,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Return mask positions and indices of random-replacement compounds."""
        use_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        random_indices: list[int] = []
        draws = torch.rand(len(selected_indices), device=device, generator=generator)
        for index, action in zip(selected_indices, draws, strict=True):
            compound = compounds[index]
            if action < mask_probability:
                use_mask[compound.row, compound.columns] = True
            elif action < mask_probability + random_probability:
                random_indices.append(index)
        return use_mask, random_indices

    @staticmethod
    def _replace_compounds_from_batch(
        categorical: torch.Tensor,
        categorical_presence: torch.Tensor,
        continuous: torch.Tensor,
        output_presence: torch.Tensor,
        original_categorical: torch.Tensor,
        original_categorical_presence: torch.Tensor,
        original_continuous: torch.Tensor,
        original_presence: torch.Tensor,
        compounds: list[_Compound],
        random_compound_indices: list[int],
        *,
        generator: torch.Generator | None,
    ) -> list[int]:
        """Copy complete, same-length compounds for semantic random actions.

        Copying a whole compound rather than sampling each field independently
        keeps NOTE/SLIDE_SEGMENT rows and their categorical/presence grammar
        valid. A donor is never the target compound itself. Returns the target
        indices for which the batch has no same-length donor (rare: long slide
        compounds); the caller downgrades those to mask actions so the
        80/10/10 action mix does not silently skew toward "keep".
        """
        if not random_compound_indices:
            return []

        unreplaced: list[int] = []
        lengths = [compound.length for compound in compounds]
        for target_index in random_compound_indices:
            candidates = [
                index
                for index, length in enumerate(lengths)
                if index != target_index and length == lengths[target_index]
            ]
            if not candidates:
                unreplaced.append(target_index)
                continue
            candidate_index = candidates[
                int(
                    torch.randint(
                        len(candidates),
                        (1,),
                        device=categorical.device,
                        generator=generator,
                    ).item()
                )
            ]
            target = compounds[target_index]
            donor = compounds[candidate_index]
            categorical[target.row, target.columns] = original_categorical[donor.row, donor.columns]
            categorical_presence[target.row, target.columns] = original_categorical_presence[
                donor.row, donor.columns
            ]
            continuous[target.row, target.columns] = original_continuous[donor.row, donor.columns]
            output_presence[target.row, target.columns] = original_presence[
                donor.row, donor.columns
            ]
        return unreplaced

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        original_categorical = batch["categorical"]
        original_continuous = batch["continuous"]
        categorical = original_categorical.clone()
        continuous = original_continuous.clone()
        original_cat_presence = batch["categorical_presence"].bool()
        cat_presence = original_cat_presence.clone()
        cont_presence = batch["continuous_presence"].bool()
        output_cont_presence = cont_presence.clone()
        valid_events = batch["attention_mask"].bool()
        generator = self._generator(batch, valid_events.device)

        supplied_note_start = batch.get("note_start")
        if supplied_note_start is None:
            raise ValueError("note_start is required for compound-level masking")
        note_start = torch.as_tensor(
            supplied_note_start, dtype=torch.bool, device=valid_events.device
        )
        if note_start.shape != valid_events.shape:
            raise ValueError("note_start must have the same shape as attention_mask")
        note_start &= valid_events

        compounds = self._compounds(valid_events, note_start)
        semantic_selected = torch.zeros_like(valid_events)
        selected_compound_indices: list[int] = []
        if compounds:
            semantic_selected, selected_compound_indices = self._select_compounds(
                compounds,
                self.probability,
                shape=valid_events.shape,
                device=valid_events.device,
                generator=generator,
            )

        # If a batch ends up targetless, add one compound with a genuine categorical target.
        has_categorical_target = bool((semantic_selected.unsqueeze(-1) & cat_presence).any())
        if compounds and not has_categorical_target:
            targetable_indices = [
                index
                for index, compound in enumerate(compounds)
                if bool(cat_presence[compound.row, compound.columns].any())
            ]
            if targetable_indices:
                choice = int(
                    torch.randint(
                        len(targetable_indices),
                        (1,),
                        device=valid_events.device,
                        generator=generator,
                    ).item()
                )
                selected_index = targetable_indices[choice]
                selected_compound_indices.append(selected_index)
                selected = compounds[selected_index]
                semantic_selected[selected.row, selected.columns] = True

        categorical_labels = torch.full_like(categorical, -100)
        categorical_label_mask = semantic_selected.unsqueeze(-1) & original_cat_presence
        categorical_labels[categorical_label_mask] = original_categorical[categorical_label_mask]

        continuous_labels = original_continuous.clone()
        continuous_label_mask = semantic_selected.unsqueeze(-1) & cont_presence

        if compounds:
            use_mask_semantic, random_compound_indices = self._compound_actions(
                compounds,
                selected_compound_indices,
                shape=valid_events.shape,
                mask_probability=self.mask_probability,
                random_probability=self.random_probability,
                device=valid_events.device,
                generator=generator,
            )
        else:
            use_mask_semantic = torch.zeros_like(valid_events)
            random_compound_indices = []

        # Corrupt every categorical slot incl. NA (NA leaks kind); random actions reuse donors.
        unreplaced_random = self._replace_compounds_from_batch(
            categorical,
            cat_presence,
            continuous,
            output_cont_presence,
            original_categorical,
            original_cat_presence,
            original_continuous,
            cont_presence,
            compounds,
            random_compound_indices,
            generator=generator,
        )
        # Random actions without a same-length donor fall back to mask actions.
        for index in unreplaced_random:
            compound = compounds[index]
            use_mask_semantic[compound.row, compound.columns] = True
        for field in range(len(self.vocab_sizes)):
            categorical[:, :, field][use_mask_semantic] = SPECIAL_TOKENS["MASK"]

        # Semantic corruption hides timing values to prevent Hold/Slide type leakage.
        continuous[use_mask_semantic] = 0.0
        output_cont_presence[use_mask_semantic] = False

        return {
            **batch,
            "categorical": categorical,
            "categorical_presence": cat_presence,
            "continuous": continuous,
            "continuous_presence": output_cont_presence,
            "categorical_labels": categorical_labels,
            "categorical_label_mask": categorical_label_mask,
            "continuous_labels": continuous_labels,
            "continuous_label_mask": continuous_label_mask,
            # Integer survives device transfer, so metrics weight without GPU sync.
            "mask_target_count": int(categorical_label_mask.sum())
            + int(continuous_label_mask.sum()),
            "semantic_masked_events": semantic_selected,
            "masked_events": semantic_selected,
        }


@dataclass(slots=True)
class MaskedEventBatchCollator:
    masking: MaskedEventCollator
    pad_to_multiple: int = 1

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self.masking(collate_events(items, pad_to_multiple=self.pad_to_multiple))
