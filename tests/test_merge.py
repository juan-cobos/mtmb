"""Tests for merge(): id-space offsetting, category consistency, image filtering."""

from __future__ import annotations

import json

import pytest

from mtmb.dataset import OFFSET, Task
from mtmb.merge import merge


def test_merge_rejects_no_tasks(tiny_dataset):
    # Through `dataset.merge`, `[] or self.tasks` falls back to every task -- the
    # empty-tasks guard only bites when called directly, as `build` never passes [].
    with pytest.raises(ValueError, match="no tasks"):
        merge(tiny_dataset, [])


def test_merge_offsets_ids_by_task_index(tiny_dataset, n_frames):
    index = Task.t_maze.get_index()
    coco = tiny_dataset.merge(["t_maze"])
    assert len(coco["images"]) == n_frames
    image_ids = {img["id"] for img in coco["images"]}
    assert image_ids == {index * OFFSET + i for i in range(1, n_frames + 1)}
    for ann in coco["annotations"]:
        assert ann["id"] >= index * OFFSET
        assert ann["image_id"] in image_ids
        assert ann["track_id"] == index * 1000  # local track_id is 0 for every instance


def test_merge_prefixes_file_name_with_task(tiny_dataset):
    coco = tiny_dataset.merge(["barnes_maze"])
    assert all(img["file_name"].startswith("barnes_maze/images/") for img in coco["images"])


def test_merge_keep_image_ids_filters_images_and_annotations(tiny_dataset):
    all_ids = tiny_dataset.image_ids("barnes_maze")
    keep = set(all_ids[:2])
    coco = tiny_dataset.merge(["barnes_maze"], keep_image_ids=keep)
    assert {img["id"] for img in coco["images"]} == keep
    assert all(ann["image_id"] in keep for ann in coco["annotations"])


def test_merge_rejects_mismatched_categories(tiny_dataset):
    # Corrupt one task's category registry after the fixture wrote it.
    path = tiny_dataset.annotations_path("nort")
    with open(path) as f:
        coco = json.load(f)
    coco["categories"][0]["name"] = "not_a_mouse"
    with open(path, "w") as f:
        json.dump(coco, f)

    with pytest.raises(ValueError, match="different categories"):
        tiny_dataset.merge(["barnes_maze", "nort"])
