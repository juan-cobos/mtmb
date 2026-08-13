"""Tests for validate_dataset: it should catch a build gone wrong, not just pass."""

from __future__ import annotations

from mtmb.validate import validate_dataset


def test_validate_dataset_passes_a_real_build(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], [], name="m")
    out_dir = tiny_dataset.build(manifest)
    report = validate_dataset(out_dir)
    assert report.ok
    assert not report.errors


def test_validate_dataset_flags_a_missing_image_file(tiny_dataset):
    manifest = tiny_dataset.split_by_task(["barnes_maze"], ["nort"], [], name="m")
    out_dir = tiny_dataset.build(manifest)
    next((out_dir / "train").glob("barnes_maze__*.jpg")).unlink()

    report = validate_dataset(out_dir)
    assert not report.ok
    assert any("not found" in e for e in report.errors)


def test_validate_dataset_flags_a_missing_train_split(tmp_path):
    report = validate_dataset(tmp_path / "empty")
    assert not report.ok
    assert any("train/" in e for e in report.errors)
