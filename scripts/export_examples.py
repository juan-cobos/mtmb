"""Encode a short, unannotated clip per task, to use as example inputs (e.g. mouselite)."""

import shutil
import subprocess
from pathlib import Path

from mtmb.dataset import MouseDataset

FPS = 30  # the frames are decoded at 30 fps, so the clip plays at real speed
SECONDS = 10
CRF = 23


def to_clip(images_dir: Path, out_path: Path, start: int, frames: int) -> Path:
    """Encode ``frames`` consecutive frames of ``images_dir``, from ``start``, as H.264.

    yuv420p and faststart so browsers can play it, and stream it before it fully loads.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-start_number",
            str(start),
            "-i",
            str(images_dir / "frame_%05d.jpg"),
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-crf",
            str(CRF),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
    )
    return out_path


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH -- required to encode clips")

    dataset = MouseDataset()
    examples_dir = dataset.build_dir / "examples"
    frames = FPS * SECONDS
    print(f"{dataset.path} -- {len(dataset.tasks)} tasks")
    for task in dataset.tasks:
        images_dir = dataset.task_dir(task) / "images"
        total = len(list(images_dir.glob("frame_*.jpg")))
        start = max(total - frames, 0) // 2  # from the middle, mice are settled by then
        out_path = to_clip(images_dir, examples_dir / f"{task.value}.mp4", start, frames)
        size_mb = out_path.stat().st_size / 1_000_000
        print(f"  {task.value:<20} -> {out_path} ({size_mb:.1f} MB)  (start={start})")
