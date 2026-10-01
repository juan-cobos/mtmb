"""The dataset itself: what tasks exist, where they live and how ids are scoped.

Everything else in the package is derived from this module. Per-task COCO files are
the canonical source of truth; merged files, split manifests and RF-DETR exports
are build artifacts.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mtmb.manifest import Manifest
    from mtmb.splits import ValidMode

DEFAULT_PATH = Path("./dataset")

#: A task's COCO annotations: boxes, segmentation and keypoints in one file.
ANNOTATIONS = "annotations.json"
SPLITS = ("train", "valid", "test")

#: Where ``download`` fetches from; task directories sit at the repo root.
HF_REPO_ID = "juancobos/MultiTaskMouseBehaviour"

#: Id space reserved per task. Larger than any plausible frame or instance count,
#: so global ids stay readable: 4_000_123 is task 4, original id 123.
OFFSET = 1_000_000


class Task(enum.StrEnum):
    """The nine assays the dataset covers; the value is the directory name.

    Members are declared in sorted order and a member's position is its task
    index, which scopes the global image, annotation and track ids that merged
    files carry. Inserting a task out of order renumbers the ones after it and
    invalidates every split manifest already saved.
    """

    barnes_maze = "barnes_maze"
    direct_interaction = "direct_interaction"
    marble = "marble"
    nort = "nort"
    nort2 = "nort2"
    open_field = "open_field"
    social_interaction = "social_interaction"
    t_maze = "t_maze"
    three_chamber = "three_chamber"

    def get_index(self) -> int:
        return TASK_INDEX[self.value]


ALL_TASKS: tuple[Task, ...] = tuple(Task)
TASK_INDEX: dict[str, int] = {task.value: i for i, task in enumerate(ALL_TASKS)}


@dataclass(frozen=True)
class MouseDataset:
    """One dataset directory and the tasks to read out of it.

    ``path`` defaults to ``./dataset``. ``build_dir`` is where exports are written,
    one directory per split, each carrying the manifest that produced it; it
    defaults beside ``path``, so ``./build`` unless ``build_root`` says otherwise.
    """

    path: Path | str = DEFAULT_PATH
    tasks: tuple[Task, ...] = field(default=ALL_TASKS)
    build_root: Path | str | None = None

    def __post_init__(self) -> None:
        # Frozen, so coerced through ``object.__setattr__``: ``"data"`` reads as
        # ``Path("data")`` everywhere ``path`` and ``build_root`` are joined.
        object.__setattr__(self, "path", Path(self.path))
        if self.build_root is not None:
            object.__setattr__(self, "build_root", Path(self.build_root))

    @property
    def build_dir(self) -> Path:
        """Where exports are written, derived from ``path`` unless overridden."""
        return self.build_root or self.path.parent / "build"

    def task(self, task: Task | str) -> Task:
        """Coerce to ``Task``, rejecting names this instance does not cover."""
        try:
            resolved = Task(task)
        except ValueError:
            raise KeyError(
                f"unknown task {task!r}; known tasks: "
                f"{', '.join(t.value for t in ALL_TASKS)}",
            ) from None
        if resolved not in self.tasks:
            raise KeyError(
                f"{resolved.value} is not part of this dataset, which covers: "
                f"{', '.join(t.value for t in self.tasks)}",
            )
        return resolved

    def task_dir(self, task: Task | str) -> Path:
        """Directory holding one task's ``images/`` and annotations."""
        return self.path / self.task(task).value

    def annotations_path(self, task: Task | str) -> Path:
        """Path to a task's annotation file."""
        if not self.path.is_dir():
            raise FileNotFoundError(f"no dataset directory at {self.path}")
        task_dir = self.task_dir(task)
        if not task_dir.is_dir():
            raise FileNotFoundError(f"{task} has no directory at {task_dir}")
        path = task_dir / ANNOTATIONS
        if not path.exists():
            raise FileNotFoundError(f"{task} has no {ANNOTATIONS}")
        return path

    def load_task(self, task: Task | str) -> dict:
        """Read a task's COCO annotations."""
        with open(self.annotations_path(task)) as f:
            return json.load(f)

    def summary(self) -> list[dict]:
        """One row per task: image, annotation and per-frame instance counts."""
        rows = []
        for task in self.tasks:
            coco = self.load_task(task)
            per_image: dict[int, int] = {}
            for ann in coco["annotations"]:
                per_image[ann["image_id"]] = per_image.get(ann["image_id"], 0) + 1
            rows.append(
                {
                    "task": task.value,
                    "index": task.get_index(),
                    "images": len(coco["images"]),
                    "annotations": len(coco["annotations"]),
                    "max_instances": max(per_image.values(), default=0),
                },
            )
        return rows

    # ---------------------------------------------------------------- id space

    def global_image_id(self, task: Task | str, image_id: int) -> int:
        return self.task(task).get_index() * OFFSET + image_id

    def global_track_id(self, task: Task | str, track_id: int) -> int:
        return self.task(task).get_index() * 1000 + track_id

    def image_ids(self, task: Task | str, every: int = 1) -> list[int]:
        """Global image ids of one task, in frame order, keeping every ``every``-th."""
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        coco = self.load_task(task)
        images = sorted(coco["images"], key=lambda im: im.get("frame_index", im["id"]))
        return [self.global_image_id(task, im["id"]) for im in images[::every]]

    def merge(
        self,
        tasks: Sequence[Task | str] | None = None,
        keep_image_ids: set[int] | None = None,
    ) -> dict:
        """Merge ``tasks`` (default: all of them) into one COCO dict."""
        from mtmb.merge import merge

        return merge(self, list(tasks or self.tasks), keep_image_ids)

    def split_by_task(
        self,
        train: Sequence[Task | str],
        valid: Sequence[Task | str],
        test: Sequence[Task | str],
        name: str = "by_task",
        every: int = 1,
    ) -> Manifest:
        """Assign whole tasks to each split -- the no-leak protocol."""
        from mtmb.splits import split_by_task

        return split_by_task(self, train, valid, test, name, every)

    def split_leave_one_out(
        self,
        tasks: Sequence[Task | str] | None = None,
        name: str = "loao",
        valid_fraction: float = 0.1,
        seed: int = 0,
        every: int = 1,
        folds: int | None = None,
        valid_mode: ValidMode | str = "in_domain",
    ) -> list[Manifest]:
        """One manifest per task, that task held out as test."""
        from mtmb.splits import split_leave_one_out

        return split_leave_one_out(
            self,
            tasks,
            name,
            valid_fraction,
            seed,
            every,
            folds,
            valid_mode,
        )

    def split_random(
        self,
        tasks: Sequence[Task | str] | None = None,
        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        name: str = "random",
        seed: int = 0,
        every: int = 1,
    ) -> Manifest:
        """Image-level random split. Leaks temporally -- read its scores as a ceiling."""
        from mtmb.splits import split_random

        return split_random(self, tasks, ratios, name, seed, every)

    def load_manifest(self, name: str) -> Manifest:
        """Read the manifest of a built export. Manifests are written by ``build``."""
        from mtmb.manifest import Manifest

        return Manifest.load(self, name)

    def manifests(self) -> list[str]:
        """Names of the exports under ``build_dir``, ready to pass to ``build``."""
        from mtmb.manifest import MANIFEST

        return sorted(p.parent.name for p in self.build_dir.glob(f"*/{MANIFEST}"))

    def build(
        self,
        manifest: Manifest | str,
        out_dir: Path | None = None,
        link: str = "symlink",
        keypoint_threshold: float = 0.3,
        overwrite: bool = False,
    ) -> Path:
        """Materialise a manifest (or a saved split's name) as an RF-DETR dataset."""
        from mtmb.splits import build

        if isinstance(manifest, str):
            manifest = self.load_manifest(manifest)
        return build(self, manifest, out_dir, link, keypoint_threshold, overwrite)

    def annotate_video(
        self,
        task: Task | str,
        out_path: Path | None = None,
        every: int = 1,
        fps: float = 30.0,
        keypoint_threshold: float | None = 0.3,
    ) -> Path:
        """Render one task's annotations over its frames as a video, to eyeball them."""
        from mtmb.video import annotate_video

        return annotate_video(self, task, out_path, every, fps, keypoint_threshold)

    def annotate_videos(
        self,
        tasks: Sequence[Task | str] | None = None,
        out_dir: Path | None = None,
        **kwargs,
    ) -> list[Path]:
        """One annotated video per task, written to ``<build_dir>/videos/``."""
        from mtmb.video import annotate_videos

        return annotate_videos(self, tasks, out_dir, **kwargs)

    def to_deeplabcut(
        self,
        manifest: Manifest | str,
        out_dir: Path | None = None,
        link: str = "symlink",
        keypoint_threshold: float = 0.3,
    ) -> Path:
        """Materialise a manifest (or a saved split's name) as a DeepLabCut project."""
        from mtmb.splits import to_deeplabcut

        if isinstance(manifest, str):
            manifest = self.load_manifest(manifest)
        return to_deeplabcut(self, manifest, out_dir, link, keypoint_threshold)

    def download(self, revision: str | None = None) -> Path:
        """Fetch this instance's tasks from the Hugging Face Hub into ``self.path``.

        Each task is its ``annotations.json`` and one ``images.tar``, unpacked into
        the ``images/`` everything else reads. The archive stays on disk as the
        Hub client's cache: a repeat call checks it against the remote and neither
        downloads nor unpacks a task that has not changed.
        """
        from huggingface_hub import snapshot_download

        from mtmb.hub import IMAGES_ARCHIVE, unpack_images

        snapshot_download(
            HF_REPO_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=self.path,
            allow_patterns=[
                f"{task.value}/{name}"
                for task in self.tasks
                for name in (ANNOTATIONS, IMAGES_ARCHIVE)
            ],
        )
        for task in self.tasks:
            task_dir = self.task_dir(task)
            if not (task_dir / IMAGES_ARCHIVE).exists():
                raise FileNotFoundError(
                    f"{HF_REPO_ID}@{revision or 'main'} has no {task.value}/"
                    f"{IMAGES_ARCHIVE}; the revision predates packed frames",
                )
            if unpack_images(task_dir):
                print(f"  unpacked {task.value}/{IMAGES_ARCHIVE}")
        return self.path


if __name__ == "__main__":  # python -m mtmb.dataset
    dataset = MouseDataset()
    print(f"{dataset.path} -- {len(dataset.tasks)} tasks")
    for row in dataset.summary():
        print(
            f"  {row['index']} {row['task']:<20} {row['images']:>6} images "
            f"{row['annotations']:>6} annotations  {row['max_instances']} per frame",
        )

    manifest = dataset.split_random(name="pooled")
    print(f"\n{manifest.describe()}")
    print(f"  built -> {dataset.build(manifest)}")
