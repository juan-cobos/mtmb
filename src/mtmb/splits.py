from __future__ import annotations

import enum
import json
import math
import random
import shutil
from collections.abc import Sequence
from pathlib import Path

from supervision.dataset.utils import train_test_split

from mtmb.dataset import SPLITS, MouseDataset, Task
from mtmb.manifest import MANIFEST, Manifest, Split
from mtmb.merge import LinkMode, link_images, merge, write
from mtmb.validate import print_report, validate_dataset


def _rethreshold_keypoints(ann: dict, threshold: float) -> dict:
    """Recompute ``v`` (and ``num_keypoints``) from ``keypoint_scores``."""
    flat = list(ann["keypoints"])
    scores = ann.get("keypoint_scores")
    if scores is None:  # nothing to re-derive from; leave the flags alone
        return {}
    for i, score in enumerate(scores):
        flat[3 * i + 2] = 2 if score >= threshold else 0
    return {
        "keypoints": flat,
        "num_keypoints": sum(1 for s in scores if s >= threshold),
    }


def to_rfdetr(
    coco: dict,
    link: LinkMode | str = LinkMode.symlink,
    keypoint_threshold: float = 0.3,
) -> dict:
    """Rewrite a merged COCO into the layout RF-DETR's loader expects.

    RF-DETR reads ``<split>/_annotations.coco.json`` with the images sitting flat
    beside it, so ``file_name`` must be a bare name. Merged paths look like
    ``t_maze/images/frame_00000.jpg``; they are flattened to ``t_maze__frame_00000.jpg``,
    which keeps them unique across tasks (every task numbers frames from 0).

    Nothing is pruned: boxes, masks and keypoints all ship, and a training run
    reads the fields its head needs. Filtering here would mean one export per head
    out of the same frames, and a detection run silently unable to tell whether the
    masks were missing or merely dropped.

    Every category gets a ``skeleton`` key, which RF-DETR's schema inference expects
    to be present even when empty. Flip pairs are inferred from the
    ``left_``/``right_`` names, so no skeleton edges are required.

    ``keypoint_threshold`` re-derives the COCO ``v`` flag from ``keypoint_scores``.
    RF-DETR treats ``v = 0`` as unlabelled, so this decides how much pose
    supervision the loss sees: at 0.3 (how the annotations were written) 26% of all
    keypoints are dropped, at 0 every keypoint is labelled. Coordinates are kept
    either way -- only the flag changes.

    The returned dict carries ``_links``: ``(source_path, flat_name)`` pairs the
    caller materialises as symlinks/hardlinks/copies, and ``_link_mode`` saying
    which. Both are dropped on write.

    Not ``sv.DetectionDataset.as_coco()``, though the split arithmetic above does
    use supervision. ``sv.Detections`` holds only ``xyxy``/``mask``/``class_id``/
    ``tracker_id``/``data``, so a COCO round-trip through it drops ``keypoints``,
    ``keypoint_scores`` and ``num_keypoints`` outright -- measured on ``nort``:
    27 points per instance in, zero out. It also flattens ``file_name`` to the
    basename, and since every task numbers frames from 0, 9828 of 11956 images
    (82%) would overwrite each other on disk while the JSON still listed them all.
    And it copies rather than links, which is 6.8 GB per split.
    """
    links: list[tuple[str, str]] = []
    images = []
    for image in coco["images"]:
        # "t_maze/images/frame_00000.jpg" -> "t_maze__frame_00000.jpg"
        flat = f"{image['task']}__{Path(image['file_name']).name}"
        links.append((image["file_name"], flat))
        images.append({**image, "file_name": flat})

    return {
        **coco,
        "licenses": [],
        "images": images,
        "annotations": [
            {**ann, **_rethreshold_keypoints(ann, keypoint_threshold)}
            for ann in coco["annotations"]
        ],
        "categories": [{"skeleton": [], **category} for category in coco["categories"]],
        "_links": links,
        "_link_mode": LinkMode(link),
    }


#: Which split each of DeepLabCut's two annotation files is cut from. It evaluates
#: on ``test.json`` *while training*, which is the role ``valid`` plays here, so
#: the manifest's own ``test`` is left out of the export and stays unseen.
DLC_SPLITS = {"train": "train", "test": "valid"}


def to_deeplabcut(
    dataset: MouseDataset,
    manifest: Manifest,
    out_dir: Path | None = None,
    link: LinkMode | str = LinkMode.symlink,
    keypoint_threshold: float = 0.3,
    splits: dict[str, str] = DLC_SPLITS,
) -> Path:
    """Materialise a manifest as the COCO project DeepLabCut's loader expects."""
    manifest.verify(dataset)
    out_dir = out_dir or dataset.build_dir / f"{manifest.name}_dlc"
    link = LinkMode(link)

    annotations_dir = out_dir / "annotations"
    images_dir = out_dir / "images"

    for dlc_name, split_name in splits.items():
        split = manifest.splits[split_name]
        coco = merge(dataset, split.tasks, keep_image_ids=set(split.image_ids))
        n_missing = len(split) - len(coco["images"])
        if n_missing:
            raise RuntimeError(
                f"{split_name}: {n_missing} manifest image ids not found in the "
                "sources -- annotations changed since the split was saved",
            )
        coco["info"] |= {"split": split_name, "manifest": manifest.name}
        coco = to_rfdetr(coco, link=link, keypoint_threshold=keypoint_threshold)

        path = write(coco, annotations_dir / f"{dlc_name}.json")
        n_linked = link_images(dataset, coco, images_dir)
        print(
            f"  {dlc_name:<5} <- {split_name:<5} {len(coco['images']):>6} images "
            f"{len(coco['annotations']):>6} annotations  ({n_linked} {link}s) -> {path}",
        )

    print(f"  manifest -> {manifest.save(out_dir)}")
    return out_dir


# --------------------------------------------------------------------------- #
# split construction
# --------------------------------------------------------------------------- #


def split_by_task(
    dataset: MouseDataset,
    train: Sequence[Task | str],
    valid: Sequence[Task | str],
    test: Sequence[Task | str],
    name: str = "by_task",
    every: int = 1,
) -> Manifest:
    """Whole tasks to each split."""
    train, valid, test = (
        [dataset.task(t).value for t in group] for group in (train, valid, test)
    )
    overlap = set(train) & set(valid) | set(train) & set(test) | set(valid) & set(test)
    if overlap:
        raise ValueError(f"tasks in more than one split: {sorted(overlap)}")
    splits = {
        n: Split(tasks, [i for t in tasks for i in dataset.image_ids(t, every)])
        for n, tasks in zip(SPLITS, (train, valid, test), strict=True)
    }
    return Manifest.make(dataset, name, "task", splits, every=every)


class ValidMode(enum.StrEnum):
    """Where a leave-one-out fold's validation frames come from.

    ``in_domain`` slices valid out of the training tasks' frames, so it measures fit.
    ``out_domain`` gives valid a task of its own, so it estimates transfer -- at the
    cost of a training task, and of selecting checkpoints on the thing test measures.
    """

    in_domain = "in_domain"
    out_domain = "out_domain"


def split_leave_one_out(
    dataset: MouseDataset,
    tasks: Sequence[Task | str] | None = None,
    name: str = "loao",
    valid_fraction: float = 0.1,
    seed: int = 0,
    every: int = 1,
    folds: int | None = None,
    valid_mode: ValidMode | str = ValidMode.in_domain,
) -> list[Manifest]:
    """One manifest per task: that task is test, the rest train (minus a valid slice).

    ``valid_mode`` decides what valid is; see ``ValidMode``. Under ``out_domain`` the
    valid task is the next one after the held-out task, wrapping around -- each task
    validates exactly once, where a random draw would skip some and move with the
    seed. ``valid_fraction`` is unused there, valid being a whole task.

    ``folds`` builds only that many folds, the held-out tasks drawn under ``seed``.
    Only the choice of folds is random: each is still trained on every task but its own.

    Image ids are read once per task and shared across folds, which every fold pools.
    """
    tasks = [dataset.task(t).value for t in tasks or dataset.tasks]
    valid_mode = ValidMode(valid_mode)
    if folds is not None and not 1 <= folds <= len(tasks):
        raise ValueError(f"folds must be between 1 and {len(tasks)}, got {folds}")
    if valid_mode is ValidMode.out_domain and len(tasks) < 3:
        raise ValueError(
            f"{valid_mode} needs at least 3 tasks -- one each to train, validate and "
            f"test on -- got {len(tasks)}: {', '.join(tasks)}",
        )
    held_out_tasks = random.Random(seed).sample(tasks, folds) if folds else tasks
    image_ids = {t: dataset.image_ids(t, every) for t in tasks}

    manifests = []
    for fold, held_out in enumerate(held_out_tasks):
        if valid_mode is ValidMode.out_domain:
            valid_task = tasks[(tasks.index(held_out) + 1) % len(tasks)]
            train_tasks = [t for t in tasks if t not in (held_out, valid_task)]
            train_ids = [i for t in train_tasks for i in image_ids[t]]
            valid_split = Split([valid_task], image_ids[valid_task])
        else:
            train_tasks = [t for t in tasks if t != held_out]
            pooled = [i for t in train_tasks for i in image_ids[t]]
            train_ids, held = train_test_split(
                pooled,
                train_ratio=1 - valid_fraction,
                random_state=seed + fold,
            )
            valid_split = Split(train_tasks, held)

        splits = {
            "train": Split(train_tasks, train_ids),
            "valid": valid_split,
            "test": Split([held_out], image_ids[held_out]),
        }
        manifests.append(
            Manifest.make(
                dataset,
                # The task, not just the index: with sampled folds, `fold_00` alone
                # does not say which assay is held out.
                f"{name}_fold_{fold:02d}_{held_out}",
                f"loo/{valid_mode.value}_valid",
                splits,
                seed + fold,
                every,
            ),
        )
    return manifests


def split_random(
    dataset: MouseDataset,
    tasks: Sequence[Task | str] | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    name: str = "random",
    seed: int = 0,
    every: int = 1,
) -> Manifest:
    """Image-level random assignment pooled across ``tasks``.

    Frames are contiguous at ~30 fps, so a random split puts near-duplicate
    neighbours on both sides of the train/test boundary. Useful as a sanity check
    or an optimistic ceiling; not a generalisation estimate.
    """
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios} = {sum(ratios)}")
    tasks = [dataset.task(t).value for t in tasks or dataset.tasks]
    pooled = [i for t in tasks for i in dataset.image_ids(t, every)]
    train_ids, rest = train_test_split(pooled, train_ratio=ratios[0], random_state=seed)
    # `rest` came back already shuffled, so the second cut does not reshuffle --
    # valid/test stay a function of the one seed rather than of a second draw.
    remainder = ratios[1] + ratios[2]
    valid_ids, test_ids = (
        train_test_split(rest, train_ratio=ratios[1] / remainder, shuffle=False)
        if remainder
        else ([], [])
    )
    splits = {
        "train": Split(tasks, train_ids),
        "valid": Split(tasks, valid_ids),
        "test": Split(tasks, test_ids),
    }
    return Manifest.make(dataset, name, "random", splits, seed, every)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def _already_built(manifest: Manifest, out_dir: Path) -> bool:
    """Whether ``out_dir`` already holds this exact split.

    Raises if it holds a *different* one: the split directories are written into
    rather than replaced, so a rebuild over another split leaves its images behind
    beside the new ones, listed by neither annotation file.
    """
    path = out_dir / MANIFEST
    if not path.exists():
        return False
    try:
        with open(path) as f:
            built = Manifest.from_dict(json.load(f))
    except (json.JSONDecodeError, KeyError, TypeError):
        raise RuntimeError(
            f"{path} is not readable as a manifest; delete {out_dir} or build with "
            "overwrite",
        ) from None
    if not manifest.matches(built):
        raise RuntimeError(
            f"{out_dir} already holds a different split ({built.describe()}); build "
            "with overwrite to replace it, or pick another name",
        )
    return True


def build(
    dataset: MouseDataset,
    manifest: Manifest,
    out_dir: Path | None = None,
    link: LinkMode | str = LinkMode.symlink,
    keypoint_threshold: float = 0.3,
    overwrite: bool = False,
) -> Path:
    """Materialise a manifest as an RF-DETR dataset directory, then validate it.

    Writes ``<out_dir>/{train,valid,test}/_annotations.coco.json`` with the images
    linked in flat beside them, which is the layout ``rfdetr`` auto-detects. The
    manifest itself is written to ``<out_dir>/manifest.json``, so the export carries
    the record of how it was split; ``rfdetr`` ignores anything outside the split
    directories. The build fails if validation does -- an unusable export should not
    sit on disk looking finished.

    An export whose manifest already matches is left alone and revalidated, so a run
    over nine folds resumes where an interrupted one stopped instead of relinking
    tens of thousands of images. ``overwrite`` deletes the directory and rebuilds it,
    which is also how a stale or mismatched export gets replaced.
    """
    manifest.verify(dataset)  # `Manifest.load` checks too; this covers hand-built ones
    out_dir = out_dir or dataset.build_dir / manifest.name
    link = LinkMode(link)

    if overwrite:
        # Gated on the manifest so this only ever deletes an export this package
        # wrote, and never asks `_already_built` -- a mismatched or unreadable one
        # is precisely what overwrite is for.
        if (out_dir / MANIFEST).exists():
            shutil.rmtree(out_dir)
    elif _already_built(manifest, out_dir):
        print(f"  already built -> {out_dir}")
        report = validate_dataset(out_dir)
        if not print_report(report, out_dir):
            raise RuntimeError(
                f"{out_dir} already holds this split but no longer validates; "
                "build with overwrite to replace it",
            )
        return out_dir

    for split_name, split in manifest.splits.items():
        coco = merge(dataset, split.tasks, keep_image_ids=set(split.image_ids))
        n_missing = len(split) - len(coco["images"])
        if n_missing:
            raise RuntimeError(
                f"{split_name}: {n_missing} manifest image ids not found in the "
                "sources -- annotations changed since the split was saved",
            )
        coco["info"] |= {"split": split_name, "manifest": manifest.name}
        coco = to_rfdetr(coco, link=link, keypoint_threshold=keypoint_threshold)

        split_dir = out_dir / split_name
        write(coco, split_dir / "_annotations.coco.json")
        n_linked = link_images(dataset, coco, split_dir)
        print(
            f"  {split_name:<6} {len(coco['images']):>6} images "
            f"{len(coco['annotations']):>6} annotations  ({n_linked} {link}s) "
            f"-> {split_dir}",
        )

    # After the splits, so a manifest on disk means all of them were written.
    print(f"  manifest -> {manifest.save(out_dir)}")

    report = validate_dataset(out_dir)
    if not print_report(report, out_dir):
        raise RuntimeError(f"{out_dir} failed validation; see errors above")
    return out_dir
