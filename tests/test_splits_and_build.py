"""Tests for the split constructors and the build/to_deeplabcut pipeline."""

from __future__ import annotations

import json

import pytest

from mtmb.merge import LinkMode
from mtmb.splits import ValidMode


def _all_ids(manifest):
    return {i for s in manifest.splits.values() for i in s.image_ids}


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #


def test_split_random_partitions_without_overlap(tiny_dataset):
    manifest = tiny_dataset.split_random(ratios=(0.5, 0.25, 0.25), name="m", seed=0)
    seen = [i for s in manifest.splits.values() for i in s.image_ids]
    assert len(seen) == len(set(seen))
    pooled = {i for t in tiny_dataset.tasks for i in tiny_dataset.image_ids(t)}
    assert _all_ids(manifest) == pooled


def test_split_random_rejects_bad_ratios(tiny_dataset):
    with pytest.raises(ValueError, match="sum to 1"):
        tiny_dataset.split_random(ratios=(0.5, 0.5, 0.5), name="m")


def test_split_by_task_assigns_whole_tasks(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], ["t_maze"], name="m")
    assert manifest.splits["train"].tasks == ["barnes_maze"]
    assert manifest.splits["valid"].tasks == ["nort"]
    assert manifest.splits["test"].tasks == ["t_maze"]
    assert set(manifest.splits["train"].image_ids) == set(
        tiny_dataset.image_ids("barnes_maze"),
    )


def test_split_by_task_rejects_task_in_two_splits(tiny_dataset):
    with pytest.raises(ValueError, match="more than one split"):
        tiny_dataset.split_by_task(["barnes_maze"], ["barnes_maze"], [], name="m")


def test_split_leave_one_out_holds_out_each_task_once(tiny_dataset):
    manifests = tiny_dataset.split_leave_one_out(name="m")
    held_out = {m.splits["test"].tasks[0] for m in manifests}
    assert held_out == {t.value for t in tiny_dataset.tasks}
    for manifest in manifests:
        test_task = manifest.splits["test"].tasks[0]
        assert test_task not in manifest.splits["train"].tasks


def test_split_leave_one_out_respects_folds(tiny_dataset):
    manifests = tiny_dataset.split_leave_one_out(name="m", folds=1, seed=0)
    assert len(manifests) == 1


def test_split_leave_one_out_out_domain_gives_valid_its_own_task(tiny_dataset):
    manifests = tiny_dataset.split_leave_one_out(name="m", valid_mode=ValidMode.out_domain)
    for manifest in manifests:
        test_task = manifest.splits["test"].tasks[0]
        valid_tasks = set(manifest.splits["valid"].tasks)
        train_tasks = set(manifest.splits["train"].tasks)
        assert valid_tasks.isdisjoint(train_tasks | {test_task})


def test_split_leave_one_out_out_domain_needs_three_tasks(tiny_dataset):
    with pytest.raises(ValueError, match="at least 3 tasks"):
        tiny_dataset.split_leave_one_out(
            tasks=["barnes_maze", "nort"],
            name="m",
            valid_mode=ValidMode.out_domain,
        )


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def test_build_writes_splits_and_validates(tiny_dataset, n_frames):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], ["t_maze"], name="m")
    out_dir = tiny_dataset.build(manifest)

    assert out_dir == tiny_dataset.build_dir / "m"
    train_ann = out_dir / "train" / "_annotations.coco.json"
    with open(train_ann) as f:
        coco = json.load(f)
    assert len(coco["images"]) == n_frames
    # File names are flattened and the images linked in flat beside the json.
    for image in coco["images"]:
        assert "/" not in image["file_name"]
        assert (out_dir / "train" / image["file_name"]).exists()
    # Every category carries a skeleton key, which RF-DETR's loader requires.
    assert coco["categories"][0]["skeleton"] == []

    assert (out_dir / "manifest.json").exists()
    assert tiny_dataset.load_manifest("m").matches(manifest)
    assert "m" in tiny_dataset.manifests()


def test_build_rethresholds_keypoints(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], [], name="m")
    out_dir = tiny_dataset.build(manifest, keypoint_threshold=0.3)
    with open(out_dir / "train" / "_annotations.coco.json") as f:
        coco = json.load(f)
    ann = coco["annotations"][0]
    visibility = ann["keypoints"][2::3]
    # Fixture alternates keypoint_scores 0.9/0.1; only the 0.9s clear 0.3.
    assert visibility.count(2) == 14
    assert ann["num_keypoints"] == 14


def test_build_is_idempotent_without_overwrite(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], [], name="m")
    first = tiny_dataset.build(manifest)
    second = tiny_dataset.build(manifest)  # should resume, not re-raise or duplicate
    assert first == second


def test_build_rejects_a_different_split_at_the_same_name(tiny_dataset):
    tiny_dataset.build(tiny_dataset.split_by_task(["barnes_maze"], ["nort"], [], name="m"))
    other = tiny_dataset.split_by_task(["nort"], ["t_maze"], [], name="m")
    with pytest.raises(RuntimeError, match="already holds a different split"):
        tiny_dataset.build(other)


def test_build_overwrite_replaces_existing_export(tiny_dataset):
    tiny_dataset.build(tiny_dataset.split_by_task(["barnes_maze"], ["nort"], [], name="m"))
    other = tiny_dataset.split_by_task(["nort"], ["t_maze"], [], name="m")
    out_dir = tiny_dataset.build(other, overwrite=True)
    assert tiny_dataset.load_manifest("m").matches(other)
    assert out_dir == tiny_dataset.build_dir / "m"


def test_build_hardlink_mode(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], [], name="m")
    out_dir = tiny_dataset.build(manifest, link=LinkMode.hardlink)
    with open(out_dir / "train" / "_annotations.coco.json") as f:
        coco = json.load(f)
    linked = out_dir / "train" / coco["images"][0]["file_name"]
    assert linked.stat().st_nlink >= 2


# --------------------------------------------------------------------------- #
# to_deeplabcut
# --------------------------------------------------------------------------- #


def test_to_deeplabcut_writes_train_and_test_from_train_and_valid(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], ["t_maze"], name="m")
    out_dir = tiny_dataset.to_deeplabcut(manifest)

    assert (out_dir / "annotations" / "train.json").exists()
    assert (out_dir / "annotations" / "test.json").exists()
    with open(out_dir / "annotations" / "test.json") as f:
        test_coco = json.load(f)
    # DLC's "test.json" is cut from the manifest's *valid* split, not its test.
    assert all(img["file_name"].startswith("nort__") for img in test_coco["images"])
