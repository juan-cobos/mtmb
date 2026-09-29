"""Convert each task's annotated video into a GIF, sized for the README."""

import random
import shutil
import subprocess
from pathlib import Path

from mtmb.dataset import MouseDataset

WIDTH = 480
FPS = 12
SECONDS = 6
SEED = 0


def probe_duration(video_path: Path) -> float:
    """Length of ``video_path`` in seconds, via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def to_gif(
    video_path: Path,
    out_path: Path,
    width: int = WIDTH,
    fps: int = FPS,
    seconds: float = SECONDS,
    start: float = 0,
) -> Path:
    """Encode ``seconds`` of ``video_path``, from ``start``, as a palette-optimised GIF.

    Two ffmpeg passes: the first builds a colour palette from the actual clip, the
    second dithers against it. A plain one-pass GIF encode defaults to a generic
    216-colour palette and looks visibly banded next to this. Trimmed to a short
    loop -- the source recordings run 30-70s, which is a 12-60 MB GIF at this size
    and unusable embedded in a README.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    palette = out_path.with_suffix(".png")
    seek = ["-ss", str(start), "-t", str(seconds), "-i", str(video_path)]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *seek,
            "-vf",
            f"{filters},palettegen",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *seek,
            "-i",
            str(palette),
            "-lavfi",
            f"{filters}[x];[x][1:v]paletteuse",
            str(out_path),
        ],
        check=True,
    )
    palette.unlink()
    return out_path


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH -- required to encode GIFs")

    dataset = MouseDataset()
    videos_dir = dataset.build_dir / "videos"
    gifs_dir = dataset.path.parent / "assets" / "gifs"
    videos = sorted(videos_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos found in {videos_dir} -- run export_videos.py first")

    rng = random.Random(SEED)
    print(f"{videos_dir} -- {len(videos)} videos")
    for video in videos:
        start = rng.uniform(0, max(probe_duration(video) - SECONDS, 0))
        out_path = to_gif(video, gifs_dir / f"{video.stem}.gif", start=start)
        size_mb = out_path.stat().st_size / 1_000_000
        print(f"  {video.stem:<20} -> {out_path} ({size_mb:.1f} MB)  (start={start:.1f}s)")
