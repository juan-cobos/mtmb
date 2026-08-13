"""A tiny synthetic dataset, on disk, mirroring the real per-task COCO schema."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mtmb.dataset import MouseDataset, Task

WIDTH, HEIGHT = 64, 64
N_FRAMES = 8

KEYPOINT_NAMES = [
    "nose",
    "left_ear",
    "right_ear",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "neck",
    "mid_back",
    "mouse_center",
    "mid_backend",
    "mid_backend2",
    "mid_backend3",
    "tail_base",
    "tail1",
    "tail2",
    "tail3",
    "tail4",
    "tail5",
    "left_shoulder",
    "left_midside",
    "left_hip",
    "right_shoulder",
    "right_midside",
    "right_hip",
    "tail_end",
    "head_midpoint",
]
CATEGORY = {
    "id": 1,
    "name": "mouse",
    "supercategory": "common-objects",
    "keypoints": KEYPOINT_NAMES,
}

#: Tasks in a fixed, non-adjacent-index order, so id-offset bugs don't hide behind 0.
TASKS: tuple[Task, ...] = (Task.barnes_maze, Task.nort, Task.t_maze)


def _write_task(task_dir: Path, n_frames: int = N_FRAMES) -> None:
    """Write ``n_frames`` real (tiny) jpgs plus a matching annotations.json."""
    images_dir = task_dir / "images"
    images_dir.mkdir(parents=True)
    images, annotations = [], []
    for i in range(n_frames):
        file_name = f"images/frame_{i:05d}.jpg"
        cv2.imwrite(str(task_dir / file_name), np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        images.append(
            {
                "id": i + 1,
                "file_name": file_name,
                "width": WIDTH,
                "height": HEIGHT,
                "frame_index": i,
            },
        )
        # Every other keypoint scores below the 0.3 default threshold, so
        # re-thresholding on build has something real to do.
        scores = [0.9 if k % 2 == 0 else 0.1 for k in range(len(KEYPOINT_NAMES))]
        keypoints = []
        for k in range(len(KEYPOINT_NAMES)):
            keypoints += [8.0 + k, 8.0 + k, 2 if scores[k] >= 0.3 else 0]
        annotations.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [4.0, 4.0, WIDTH - 8.0, HEIGHT - 8.0],
                "area": (WIDTH - 8.0) * (HEIGHT - 8.0),
                "segmentation": [
                    [4, 4, WIDTH - 4, 4, WIDTH - 4, HEIGHT - 4, 4, HEIGHT - 4],
                ],
                "iscrowd": 0,
                "score": 0.9,
                "track_id": 0,
                "keypoints": keypoints,
                "keypoint_scores": scores,
                "num_keypoints": sum(1 for s in scores if s >= 0.3),
            },
        )
    coco = {
        "info": {"description": "synthetic fixture"},
        "images": images,
        "annotations": annotations,
        "categories": [CATEGORY],
    }
    with open(task_dir / "annotations.json", "w") as f:
        json.dump(coco, f)


@pytest.fixture
def tasks() -> tuple[Task, ...]:
    return TASKS


@pytest.fixture
def n_frames() -> int:
    return N_FRAMES


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> MouseDataset:
    """A ``MouseDataset`` over 3 real tasks, 8 frames each, written to ``tmp_path``."""
    root = tmp_path / "dataset"
    for task in TASKS:
        _write_task(root / task.value)
    return MouseDataset(root, tasks=TASKS, build_root=tmp_path / "build")
