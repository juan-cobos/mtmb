"""Tests for MouseDataset: task coercion, paths, id-space arithmetic, summary."""

from __future__ import annotations

import pytest

from mtmb.dataset import ALL_TASKS, OFFSET, MouseDataset, Task


def test_task_index_matches_declaration_order():
    for i, task in enumerate(ALL_TASKS):
        assert task.get_index() == i


def test_task_coerces_string_and_rejects_unknown(tiny_dataset: MouseDataset):
    assert tiny_dataset.task("barnes_maze") is Task.barnes_maze
    with pytest.raises(KeyError):
        tiny_dataset.task("not_a_task")


def test_task_rejects_task_outside_this_instance(tiny_dataset: MouseDataset):
    # `open_field` is a real Task, just not one this fixture's dataset covers.
    with pytest.raises(KeyError):
        tiny_dataset.task(Task.open_field)


def test_task_dir_and_annotations_path(tiny_dataset: MouseDataset):
    task_dir = tiny_dataset.task_dir("barnes_maze")
    assert task_dir == tiny_dataset.path / "barnes_maze"
    assert tiny_dataset.annotations_path("barnes_maze") == task_dir / "annotations.json"


def test_annotations_path_missing_dataset_dir(tmp_path):
    dataset = MouseDataset(tmp_path / "nowhere", tasks=(Task.barnes_maze,))
    with pytest.raises(FileNotFoundError):
        dataset.annotations_path("barnes_maze")


def test_load_task_reads_expected_schema(tiny_dataset: MouseDataset, n_frames: int):
    coco = tiny_dataset.load_task("barnes_maze")
    assert len(coco["images"]) == n_frames
    assert len(coco["annotations"]) == n_frames
    assert coco["categories"][0]["name"] == "mouse"


def test_summary_counts_per_task(
    tiny_dataset: MouseDataset,
    tasks: tuple[Task, ...],
    n_frames: int,
):
    rows = {row["task"]: row for row in tiny_dataset.summary()}
    assert set(rows) == {t.value for t in tasks}
    for task in tasks:
        row = rows[task.value]
        assert row["images"] == n_frames
        assert row["annotations"] == n_frames
        assert row["max_instances"] == 1
        assert row["index"] == task.get_index()


def test_global_image_and_track_id_offset_by_task_index(tiny_dataset: MouseDataset):
    index = Task.t_maze.get_index()
    assert tiny_dataset.global_image_id("t_maze", 5) == index * OFFSET + 5
    assert tiny_dataset.global_track_id("t_maze", 2) == index * 1000 + 2


def test_image_ids_are_frame_ordered_and_globally_unique(
    tiny_dataset: MouseDataset,
    n_frames: int,
):
    ids = tiny_dataset.image_ids("barnes_maze")
    assert len(ids) == n_frames
    assert ids == sorted(ids)
    assert len(set(ids)) == n_frames


def test_image_ids_every_keeps_every_nth(tiny_dataset: MouseDataset):
    every = tiny_dataset.image_ids("barnes_maze", every=3)
    assert every == tiny_dataset.image_ids("barnes_maze")[::3]


def test_image_ids_rejects_bad_every(tiny_dataset: MouseDataset):
    with pytest.raises(ValueError, match="every"):
        tiny_dataset.image_ids("barnes_maze", every=0)
