"""Tests for the Hub archive format: deterministic packing, safe unpacking."""

from __future__ import annotations

from pathlib import Path

from mtmb.dataset import MouseDataset
from mtmb.hub import UNPACKED_STAMP, pack_images, unpack_images


def test_pack_is_byte_identical_for_the_same_frames(tmp_path: Path, tiny_dataset):
    task_dir = tiny_dataset.task_dir(tiny_dataset.tasks[0])
    first = pack_images(task_dir, tmp_path / "a.tar").read_bytes()
    for frame in (task_dir / "images").iterdir():
        frame.touch()  # new mtimes must not leak into the archive
    assert pack_images(task_dir, tmp_path / "b.tar").read_bytes() == first


def test_unpack_round_trips_and_repacks_without_the_stamp(
    tmp_path: Path, tiny_dataset: MouseDataset
):
    source = tiny_dataset.task_dir(tiny_dataset.tasks[0])
    task_dir = tmp_path / "task"
    pack_images(source, task_dir / "images.tar")
    (task_dir / "annotations.json").write_bytes((source / "annotations.json").read_bytes())

    assert unpack_images(task_dir)
    assert (task_dir / "images" / UNPACKED_STAMP).exists()
    assert not unpack_images(task_dir)  # same archive: nothing to do

    repacked = pack_images(task_dir, tmp_path / "again.tar").read_bytes()
    assert repacked == (task_dir / "images.tar").read_bytes()
    assert not [p for p in task_dir.iterdir() if p.name.startswith(".unpack-")]


def test_pack_ships_only_the_frames_the_annotations_list(tmp_path: Path, tiny_dataset):
    import tarfile

    task_dir = tiny_dataset.task_dir(tiny_dataset.tasks[0])
    stray = task_dir / "images" / "frame_99999.jpg"
    stray.write_bytes((task_dir / "images" / "frame_00000.jpg").read_bytes())
    with tarfile.open(pack_images(task_dir, tmp_path / "a.tar")) as tar:
        names = tar.getnames()
    assert "images/frame_99999.jpg" not in names
    assert len(names) == len(tiny_dataset.load_task(tiny_dataset.tasks[0])["images"])
