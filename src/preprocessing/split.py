from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from preprocessing.discovery import ChartKey, ChartRecord

SplitName = Literal["train", "validation", "test"]
Splits: tuple[SplitName, ...] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    group_splits: dict[str, SplitName]
    supervised_keys: dict[ChartKey, SplitName]
    pretraining_keys: frozenset[ChartKey]

    def assert_valid(self, charts: list[ChartRecord]) -> None:
        split_groups: dict[SplitName, set[str]] = {name: set() for name in Splits}
        chart_by_key = {chart.key: chart for chart in charts}
        for key, split in self.supervised_keys.items():
            split_groups[split].add(chart_by_key[key].music_id)
        if split_groups["train"] & split_groups["validation"]:
            raise AssertionError("train/validation group leakage")
        if split_groups["train"] & split_groups["test"]:
            raise AssertionError("train/test group leakage")
        if split_groups["validation"] & split_groups["test"]:
            raise AssertionError("validation/test group leakage")
        excluded = split_groups["validation"] | split_groups["test"]
        contaminated = {
            chart.music_id
            for chart in charts
            if chart.key in self.pretraining_keys and chart.music_id in excluded
        }
        if contaminated:
            raise AssertionError(f"strict pretraining contamination: {sorted(contaminated)[:5]}")


def difficulty_bin(value: float, width: float) -> int:
    return math.floor((value + 1e-10) / width)


def _group_features(records: list[ChartRecord], threshold: float, bin_width: float) -> Counter[str]:
    features: Counter[str] = Counter({"charts": len(records)})
    for record in records:
        assert record.difficulty_const is not None
        if record.difficulty_const >= threshold:
            features[f"high:{difficulty_bin(record.difficulty_const, bin_width)}"] += 1
        else:
            features["background"] += 1
    return features


def split_supervised_groups(
    charts: list[ChartRecord],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    threshold: float = 12.0,
    bin_width: float = 0.1,
    rare_bin_max_groups: int = 2,
) -> SplitAssignment:
    labeled_by_group: dict[str, list[ChartRecord]] = defaultdict(list)
    for chart in charts:
        if chart.difficulty_const is not None:
            labeled_by_group[chart.music_id].append(chart)
    if not labeled_by_group:
        raise ValueError("no labeled charts available for splitting")

    features = {
        group: _group_features(records, threshold, bin_width)
        for group, records in labeled_by_group.items()
    }
    totals: Counter[str] = Counter()
    high_bin_groups: dict[str, set[str]] = defaultdict(set)
    for group, vector in features.items():
        totals.update(vector)
        for key in vector:
            if key.startswith("high:"):
                high_bin_groups[key].add(group)
    forced_train = set().union(
        *(groups for groups in high_bin_groups.values() if len(groups) <= rare_bin_max_groups),
        set(),
    )

    ratio_by_split = dict(zip(Splits, ratios, strict=True))
    assigned: dict[str, SplitName] = {group: "train" for group in forced_train}
    current = {name: Counter[str]() for name in Splits}
    for group in forced_train:
        current["train"].update(features[group])

    rng = random.Random(seed)
    random_keys = {group: rng.random() for group in labeled_by_group}

    def rarity(group: str) -> float:
        score = 0.0
        for feature, count in features[group].items():
            if feature.startswith("high:"):
                score = max(score, count / max(totals[feature], 1))
        return score

    remaining = sorted(
        set(labeled_by_group).difference(forced_train),
        key=lambda group: (-rarity(group), -features[group]["charts"], random_keys[group]),
    )

    def objective(candidate_group: str, candidate_split: SplitName) -> float:
        score = 0.0
        for split in Splits:
            for feature, total in totals.items():
                value = current[split][feature]
                if split == candidate_split:
                    value += features[candidate_group][feature]
                target = total * ratio_by_split[split]
                weight = 2.0 if feature.startswith("high:") else 1.0
                score += weight * ((value - target) / max(target, 1.0)) ** 2
        return score

    for group in remaining:
        split = min(Splits, key=lambda name: (objective(group, name), Splits.index(name)))
        assigned[group] = split
        current[split].update(features[group])

    supervised = {
        chart.key: assigned[chart.music_id]
        for chart in charts
        if chart.difficulty_const is not None
    }
    held_out_groups = {
        group for group, split in assigned.items() if split in {"validation", "test"}
    }
    pretraining = frozenset(chart.key for chart in charts if chart.music_id not in held_out_groups)
    result = SplitAssignment(assigned, supervised, pretraining)
    result.assert_valid(charts)
    return result
