"""Tests for MouseDataset: task coercion, paths, id-space arithmetic, summary."""

from __future__ import annotations

import json

import pytest

from mtmb.dataset import ALL_TASKS, HF_REPO_ID, OFFSET, MouseDataset, Task


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


def test_download_fetches_annotations_and_one_archive_per_task(fake_hub, tiny_dataset):
    _, calls, local = fake_hub
    assert local.download(revision="v1") == local.path
    (call,) = calls
    assert call["repo_id"] == HF_REPO_ID
    assert call["repo_type"] == "dataset"
    assert call["revision"] == "v1"
    assert call["local_dir"] == local.path
    assert call["allow_patterns"] == [
        f"{task.value}/{name}"
        for task in tiny_dataset.tasks
        for name in ("annotations.json", "images.tar")
    ]
    for task in tiny_dataset.tasks:
        source = tiny_dataset.task_dir(task) / "images"
        unpacked = local.task_dir(task) / "images"
        assert sorted(p.name for p in source.iterdir()) == sorted(
            p.name for p in unpacked.iterdir() if not p.name.startswith(".")
        )
        assert local.load_task(task) == tiny_dataset.load_task(task)


def test_download_again_leaves_unchanged_tasks_alone(fake_hub, capsys):
    _, _, local = fake_hub
    local.download()
    capsys.readouterr()
    local.download()
    assert "unpacked" not in capsys.readouterr().out


def test_download_replaces_a_changed_task_whole(fake_hub, tiny_dataset):
    from mtmb.hub import pack_images

    remote, _, local = fake_hub
    local.download()
    task = tiny_dataset.tasks[0]
    # Dropped upstream: out of the annotations, then off disk.
    coco = tiny_dataset.load_task(task)
    (dropped,) = [im for im in coco["images"] if im["file_name"].endswith("_00000.jpg")]
    coco["images"].remove(dropped)
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != dropped["id"]]
    tiny_dataset.annotations_path(task).write_text(json.dumps(coco))
    (tiny_dataset.task_dir(task) / dropped["file_name"]).unlink()
    pack_images(tiny_dataset.task_dir(task), remote / task.value / "images.tar")

    local.download()
    assert not (local.task_dir(task) / "images" / "frame_00000.jpg").exists()
    assert (local.task_dir(task) / "images" / "frame_00001.jpg").exists()


def test_download_rejects_a_revision_without_archives(fake_hub, tiny_dataset):
    remote, _, local = fake_hub
    (remote / tiny_dataset.tasks[0].value / "images.tar").unlink()
    with pytest.raises(FileNotFoundError, match="predates packed frames"):
        local.download()
