"""Draw a task's own annotations back onto its frames and write them out as a video."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from mtmb.dataset import MouseDataset, Task

#: Frames were extracted from ~30 fps footage; the source videos are not shipped.
FPS = 30.0

#: One colour per track, so a mouse keeps its colour for the whole video.
COLORS = sv.ColorPalette.DEFAULT
TRACK = sv.ColorLookup.TRACK

#: Built once rather than per frame: config, not per-frame data.
MASK_ANNOTATOR = sv.MaskAnnotator(color=COLORS, opacity=0.4, color_lookup=TRACK)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLORS, thickness=2, color_lookup=TRACK)
LABEL_ANNOTATOR = sv.LabelAnnotator(color=COLORS, text_scale=0.4, color_lookup=TRACK)


def decode_mask(
    segmentation: dict | list,
    resolution_wh: tuple[int, int],
) -> np.ndarray:
    """Decode one COCO segmentation -- compressed RLE or polygons -- to a bool mask."""
    if isinstance(segmentation, dict):
        return sv.rle_to_mask(segmentation["counts"], resolution_wh)
    # Both forms occur in the same file: SAM 3 writes RLE for the shapes its
    # polygoniser cannot express, polygons for the rest.
    mask = np.zeros(resolution_wh[::-1], dtype=bool)
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        mask |= sv.polygon_to_mask(points.astype(int), resolution_wh).astype(bool)
    return mask


def to_detections(
    annotations: Sequence[dict],
    resolution_wh: tuple[int, int],
    masks: bool = True,
) -> sv.Detections:
    """One frame's COCO annotations as ``sv.Detections``, tracker ids kept."""
    if not annotations:
        return sv.Detections.empty()
    x, y, w, h = np.asarray([ann["bbox"] for ann in annotations], dtype=np.float32).T
    return sv.Detections(
        xyxy=np.stack([x, y, x + w, y + h], axis=1),
        mask=(
            np.stack(
                [decode_mask(a["segmentation"], resolution_wh) for a in annotations],
            )
            if masks
            else None
        ),
        confidence=np.asarray(
            [ann.get("score", 1.0) for ann in annotations],
            dtype=np.float32,
        ),
        class_id=np.asarray([ann["category_id"] for ann in annotations], dtype=int),
        tracker_id=np.asarray(
            [ann.get("track_id", 0) for ann in annotations],
            dtype=int,
        ),
    )


def to_keypoints(annotations: Sequence[dict], threshold: float | None) -> sv.KeyPoints:
    """One frame's COCO keypoints as ``sv.KeyPoints``, in annotation order.

    ``threshold`` re-derives visibility from ``keypoint_scores`` the way the exports
    do; ``None`` trusts the ``v`` flags already in the file.
    """
    xy, visible, scores = [], [], []
    for ann in annotations:
        points = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        score = ann.get("keypoint_scores")
        score = (
            np.asarray(score, dtype=np.float32)
            if score is not None
            else points[:, 2] / 2  # no scores to threshold on: v=2 -> 1.0, v=0 -> 0.0
        )
        xy.append(points[:, :2])
        visible.append(
            score >= threshold if threshold is not None else points[:, 2] > 0,
        )
        scores.append(score)
    return sv.KeyPoints(
        xy=np.stack(xy),
        class_id=np.asarray([ann["category_id"] for ann in annotations], dtype=int),
        keypoint_confidence=np.stack(scores),
        visible=np.stack(visible),
    )


def annotate_frame(
    frame: np.ndarray,
    annotations: Sequence[dict],
    names: dict[int, str],
    keypoint_threshold: float | None = 0.3,
) -> np.ndarray:
    """Draw one frame's annotations onto it, in place."""
    if not annotations:
        return frame
    resolution_wh = (frame.shape[1], frame.shape[0])
    detections = to_detections(annotations, resolution_wh)

    MASK_ANNOTATOR.annotate(frame, detections)
    BOX_ANNOTATOR.annotate(frame, detections)
    LABEL_ANNOTATOR.annotate(
        frame,
        detections,
        labels=[
            f"{names.get(ann['category_id'], ann['category_id'])} "
            f"#{ann.get('track_id', 0)} {ann.get('score', 1.0):.2f}"
            for ann in annotations
        ],
    )
    points = to_keypoints(annotations, keypoint_threshold)
    # One instance at a time: VertexAnnotator draws a whole KeyPoints in a single
    # colour, and the point of the colour is to say which mouse it belongs to.
    for i, tracker_id in enumerate(detections.tracker_id):
        sv.VertexAnnotator(color=COLORS.by_idx(int(tracker_id)), radius=3).annotate(
            frame,
            points[i : i + 1],
        )
    return frame


def annotate_video(
    dataset: MouseDataset,
    task: Task | str,
    out_path: Path | None = None,
    every: int = 1,
    fps: float = FPS,
    keypoint_threshold: float | None = 0.3,
) -> Path:
    """Render one task's frames, annotated, to ``<build_dir>/videos/<task>.mp4``.

    Written under ``build_dir`` rather than beside the frames, since it is a build
    artifact: the task directories hold sources only. ``every`` subsamples frames
    without slowing the playback, so a check video of a long task can be shorter
    than the footage it came from.
    """
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    task = dataset.task(task)
    coco = dataset.load_task(task)
    task_dir = dataset.task_dir(task)

    images = sorted(coco["images"], key=lambda im: im.get("frame_index", im["id"]))[::every]
    if not images:
        raise ValueError(f"{task} has no images to render")
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    names = {c["id"]: c["name"] for c in coco["categories"]}

    out_path = out_path or dataset.build_dir / "videos" / f"{task.value}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = sv.VideoInfo(
        width=images[0]["width"],
        height=images[0]["height"],
        fps=fps,
        total_frames=len(images),
    )

    n_annotations = 0
    with sv.VideoSink(str(out_path), info) as sink:
        for image in images:
            path = task_dir / image["file_name"]
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"{task}: cannot read frame {path}")
            if (frame.shape[1], frame.shape[0]) != (info.width, info.height):
                raise ValueError(
                    f"{task}: {path} is {frame.shape[1]}x{frame.shape[0]}, but the "
                    f"video is {info.width}x{info.height} -- frames must share a size",
                )
            annotations = by_image.get(image["id"], [])
            n_annotations += len(annotations)
            sink.write_frame(
                annotate_frame(frame, annotations, names, keypoint_threshold),
            )

    print(
        f"  {task.value:<20} {len(images):>6} frames {n_annotations:>6} annotations "
        f"-> {out_path}",
    )
    return out_path


def annotate_videos(
    dataset: MouseDataset,
    tasks: Sequence[Task | str] | None = None,
    out_dir: Path | None = None,
    **kwargs,
) -> list[Path]:
    """Render one annotated video per task (default: every task of the dataset)."""
    out_dir = out_dir or dataset.build_dir / "videos"
    return [
        annotate_video(
            dataset,
            task,
            out_path=out_dir / f"{dataset.task(task).value}.mp4",
            **kwargs,
        )
        for task in (tasks or dataset.tasks)
    ]
