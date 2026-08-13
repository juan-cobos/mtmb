from __future__ import annotations

import json
from pathlib import Path

from mtmb.dataset import SPLITS

RFDETR_ANNOTATIONS = "_annotations.coco.json"


class Report:
    """Collected problems, split into errors (fatal) and warnings (suspicious)."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: Report, prefix: str) -> None:
        self.errors += [f"{prefix}: {e}" for e in other.errors]
        self.warnings += [f"{prefix}: {w}" for w in other.warnings]


def _check_images(coco: dict, split_dir: Path, report: Report) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    seen_names: dict[str, int] = {}
    missing = 0
    for image in coco["images"]:
        image_id = image["id"]
        if image_id in by_id:
            report.error(f"duplicate image id {image_id}")
        by_id[image_id] = image

        name = image["file_name"]
        if "/" in name or "\\" in name:
            report.error(
                f"image {image_id} file_name {name!r} is a path; RF-DETR expects a "
                "bare filename with the image flat in the split directory",
            )
        if name in seen_names:
            report.error(f"duplicate file_name {name!r}")
        seen_names[name] = image_id

        if image.get("width", 0) <= 0 or image.get("height", 0) <= 0:
            report.error(f"image {image_id} has non-positive width/height")

        path = split_dir / name
        # A broken symlink is not `exists()`, which is exactly the failure mode of
        # a build directory whose dataset/ moved.
        if not path.exists():
            missing += 1
            if missing <= 3:
                report.error(f"image file not found: {path}")
    if missing > 3:
        report.error(f"...and {missing - 3} more missing image files")
    return by_id


def _check_bbox(ann: dict, image: dict | None, report: Report) -> None:
    bbox = ann.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        report.error(f"annotation {ann['id']} bbox is not [x, y, w, h]")
        return
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        report.error(f"annotation {ann['id']} has non-positive bbox extent {w}x{h}")
    if image is not None:
        # 1px slack: masks rounded to integer bounds can land exactly on the edge.
        box_outside_image = (
            x < -1 or y < -1 or x + w > image["width"] + 1 or y + h > image["height"] + 1
        )
        if box_outside_image:
            report.warn(f"annotation {ann['id']} bbox extends outside the image")


def _check_segmentation(ann: dict, image: dict | None, report: Report) -> None:
    segmentation = ann.get("segmentation")
    if segmentation is None:
        report.error(f"annotation {ann['id']} has no segmentation")
        return
    if isinstance(segmentation, dict):
        counts, size = segmentation.get("counts"), segmentation.get("size")
        if not isinstance(counts, (str, bytes, list)):
            report.error(
                f"annotation {ann['id']} RLE counts is "
                f"{type(counts).__name__}; RF-DETR accepts str, bytes or list",
            )
        if image is not None and size != [image["height"], image["width"]]:
            report.error(
                f"annotation {ann['id']} RLE size {size} does not match the image "
                f"[{image['height']}, {image['width']}]",
            )
        return
    polygons = [p for p in segmentation if len(p) >= 6]
    if not polygons:
        report.error(f"annotation {ann['id']} has an empty/degenerate polygon list")
    if any(len(p) % 2 for p in polygons):
        report.error(
            f"annotation {ann['id']} has a polygon with an odd coordinate count",
        )


def _check_keypoints(ann: dict, n_keypoints: int, report: Report) -> None:
    keypoints = ann.get("keypoints")
    if not keypoints:
        report.error(f"annotation {ann['id']} has no keypoints")
        return
    if len(keypoints) != 3 * n_keypoints:
        report.error(
            f"annotation {ann['id']} has {len(keypoints)} keypoint values, "
            f"expected 3 x {n_keypoints}",
        )
        return
    visibility = keypoints[2::3]
    if any(v not in (0, 1, 2) for v in visibility):
        report.error(f"annotation {ann['id']} has a visibility flag outside {{0,1,2}}")
    declared = ann.get("num_keypoints")
    if declared is None:
        report.error(f"annotation {ann['id']} is missing num_keypoints")
    elif declared != sum(1 for v in visibility if v > 0):
        report.error(
            f"annotation {ann['id']} num_keypoints={declared} disagrees with its "
            "visibility flags",
        )


def validate_split(split_dir: Path) -> Report:
    """Validate one ``train``/``valid``/``test`` directory."""
    report = Report()
    ann_path = split_dir / RFDETR_ANNOTATIONS
    if not ann_path.exists():
        report.error(f"missing {ann_path}")
        return report
    try:
        with open(ann_path) as f:
            coco = json.load(f)
    except json.JSONDecodeError as exc:
        report.error(f"{ann_path} is not valid JSON: {exc}")
        return report

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            report.error(f"missing top-level {key!r}")
    if not report.ok:
        return report

    if not coco["images"]:
        report.error("no images")
    if not coco["annotations"]:
        report.warn("no annotations in this split")

    images = _check_images(coco, split_dir, report)

    categories = {c["id"]: c for c in coco["categories"]}
    n_keypoints = 0
    for category in coco["categories"]:
        names = category.get("keypoints")
        if not names:
            report.error(f"category {category['id']} declares no keypoints")
            continue
        if "skeleton" not in category:
            report.error(f"category {category['id']} is missing a skeleton field")
        n_keypoints = max(n_keypoints, len(names))

    seen_ids: set[int] = set()
    for ann in coco["annotations"]:
        ann_id = ann.get("id")
        if ann_id in seen_ids:
            report.error(f"duplicate annotation id {ann_id}")
        seen_ids.add(ann_id)

        image = images.get(ann["image_id"])
        if image is None:
            report.error(
                f"annotation {ann_id} references unknown image {ann['image_id']}",
            )
        if ann.get("category_id") not in categories:
            report.error(
                f"annotation {ann_id} has unknown category {ann.get('category_id')}",
            )
        if ann.get("iscrowd", 0) not in (0, 1):
            report.error(f"annotation {ann_id} has iscrowd={ann.get('iscrowd')}")
        if ann.get("area", 0) <= 0:
            report.warn(f"annotation {ann_id} has non-positive area")

        _check_bbox(ann, image, report)
        _check_segmentation(ann, image, report)
        _check_keypoints(ann, n_keypoints, report)

    annotated = {a["image_id"] for a in coco["annotations"]}
    empty = len(images) - len(annotated & set(images))
    if empty:
        report.warn(f"{empty} images have no annotations (background-only frames)")
    return report


def validate_dataset(dataset_dir: Path) -> Report:
    """Validate a whole export: RF-DETR's format probe plus every split present."""
    report = Report()
    # This is exactly what rfdetr.datasets.coco.is_valid_coco_dataset checks.
    if not (dataset_dir / "train" / RFDETR_ANNOTATIONS).exists():
        report.error(
            f"{dataset_dir} is not a COCO dataset RF-DETR can detect: "
            f"train/{RFDETR_ANNOTATIONS} is missing",
        )
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            if split != "test":
                report.error(f"missing required split directory {split_dir}")
            continue
        report.extend(validate_split(split_dir), split)
    return report


def print_report(report: Report, dataset_dir: Path) -> bool:
    for warning in report.warnings:
        print(f"  WARN  {warning}")
    for error in report.errors:
        print(f"  ERROR {error}")
    status = "OK" if report.ok else "FAILED"
    print(
        f"  {status}: {dataset_dir} "
        f"({len(report.errors)} errors, {len(report.warnings)} warnings)",
    )
    return report.ok
