"""Tests for Manifest: construction invariants, verify(), serialization."""

from __future__ import annotations

import pytest

from mtmb.dataset import Task
from mtmb.manifest import Manifest, Split


def test_make_rejects_overlapping_image_ids(tiny_dataset):
    splits = {
        "train": Split(["barnes_maze"], [1, 2, 3]),
        "valid": Split(["barnes_maze"], [3, 4]),
    }
    with pytest.raises(ValueError, match="both"):
        Manifest.make(tiny_dataset, "m", "mode", splits)


def test_make_rejects_unknown_split_name(tiny_dataset):
    splits = {"bogus": Split(["barnes_maze"], [1])}
    with pytest.raises(ValueError, match="unknown split"):
        Manifest.make(tiny_dataset, "m", "mode", splits)


def test_make_drops_empty_splits(tiny_dataset):
    splits = {
        "train": Split(["barnes_maze"], [1, 2]),
        "test": Split([], []),
    }
    manifest = Manifest.make(tiny_dataset, "m", "mode", splits)
    assert "test" not in manifest.splits
    assert manifest.tasks == ["barnes_maze"]


def test_roundtrip_to_dict_from_dict(tiny_dataset):
    splits = {"train": Split(["barnes_maze"], [2, 1])}
    manifest = Manifest.make(tiny_dataset, "m", "mode", splits, seed=7, every=2)
    restored = Manifest.from_dict(manifest.to_dict())
    assert restored.matches(manifest)
    # Sorted on construction, regardless of insertion order.
    assert restored.splits["train"].image_ids == [1, 2]


def test_verify_passes_for_a_fresh_manifest(tiny_dataset):
    manifest = tiny_dataset.split_random(name="m")
    manifest.verify(tiny_dataset)  # must not raise


def test_verify_detects_stale_task_index(tiny_dataset):
    manifest = tiny_dataset.split_random(name="m")
    stale = manifest.to_dict()
    stale["task_index"] = {t: i + 1 for t, i in stale["task_index"].items()}
    with pytest.raises(ValueError, match="task index"):
        Manifest.from_dict(stale).verify(tiny_dataset)


def test_verify_detects_annotations_mismatch(tiny_dataset):
    manifest = tiny_dataset.split_random(name="m")
    stale = manifest.to_dict()
    stale["annotations"] = "other.json"
    with pytest.raises(ValueError, match=r"other\.json"):
        Manifest.from_dict(stale).verify(tiny_dataset)


def test_matches_ignores_created_timestamp(tiny_dataset):
    splits = {"train": Split(["barnes_maze"], [1])}
    a = Manifest.make(tiny_dataset, "m", "mode", splits)
    b = Manifest.make(tiny_dataset, "m", "mode", splits)
    assert a.matches(b)


def test_describe_lists_each_split(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], ["t_maze"], name="m")
    description = manifest.describe()
    assert "m [task]" in description
    for task in (Task.barnes_maze, Task.nort, Task.t_maze):
        assert task.value in description
