"""Merge per-task COCO files into one, with globally unique ids."""

from __future__ import annotations

import enum
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from mtmb.dataset import ANNOTATIONS, OFFSET, MouseDataset, Task


class LinkMode(enum.StrEnum):
    """How images are placed beside a split's annotations."""

    symlink = "symlink"
    hardlink = "hardlink"
    copy = "copy"


def merge(
    dataset: MouseDataset,
    tasks: Sequence[Task | str],
    keep_image_ids: set[int] | None = None,
) -> dict:
    """Merge ``tasks`` into one COCO dict with globally unique ids.

    ``keep_image_ids`` filters by *global* image id (what a split manifest stores),
    keeping only those images and their annotations. ``None`` keeps everything.
    """
    if not tasks:
        raise ValueError("no tasks to merge")

    images: list[dict] = []
    annotations: list[dict] = []
    categories: list[dict] | None = None
    resolved = [dataset.task(t) for t in tasks]

    for task in resolved:
        coco = dataset.load_task(task)
        if categories is None:
            categories = coco["categories"]
        elif coco["categories"] != categories:
            raise ValueError(
                f"{task} has different categories than the other tasks; the merged "
                "file needs one shared category registry",
            )

        index = task.get_index()
        kept: set[int] = set()
        for image in coco["images"]:
            image_id = dataset.global_image_id(task, image["id"])
            if keep_image_ids is not None and image_id not in keep_image_ids:
                continue
            kept.add(image["id"])
            images.append(
                {
                    **image,
                    "id": image_id,
                    "file_name": f"{task.value}/{image['file_name']}",
                    "video_id": index,
                    "task": task.value,
                },
            )

        for ann in coco["annotations"]:
            if ann["image_id"] not in kept:
                continue
            merged = {
                **ann,
                "id": index * OFFSET + ann["id"],
                "image_id": dataset.global_image_id(task, ann["image_id"]),
            }
            if "track_id" in ann:
                merged["track_id"] = dataset.global_track_id(task, ann["track_id"])
            annotations.append(merged)

    return {
        "info": {
            "description": "Multi-task mouse dataset",
            "tasks": [t.value for t in resolved],
            "annotations": ANNOTATIONS,
            "task_index": {t.value: t.get_index() for t in resolved},
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def write(coco: dict, out_path: Path) -> Path:
    """Write a COCO dict, dropping the private ``_``-prefixed build keys."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in coco.items() if not k.startswith("_")}, f)
    return out_path


def link_file(src: Path, dst: Path, mode: LinkMode | str = LinkMode.symlink) -> None:
    """Place one image at ``dst``, replacing whatever is already there."""
    mode = LinkMode(mode)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode is LinkMode.symlink:
        dst.symlink_to(src)
    elif mode is LinkMode.hardlink:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def link_images(dataset: MouseDataset, coco: dict, split_dir: Path) -> int:
    """Materialise ``_links`` beside the annotations, as RF-DETR expects.

    Symlinks by default: the images are 6.8 GB and every split of every fold would
    otherwise duplicate them. Hardlinks are the alternative when the training host
    resolves symlinks awkwardly; copying is the last resort.
    """
    mode = LinkMode(coco.get("_link_mode", LinkMode.symlink))
    split_dir.mkdir(parents=True, exist_ok=True)
    for source, flat in coco["_links"]:
        link_file((dataset.path / source).resolve(), split_dir / flat, mode)
    return len(coco["_links"])
