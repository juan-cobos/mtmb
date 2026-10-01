"""How a task's frames travel to and from the Hugging Face Hub: one tar per task.

Fourteen thousand loose JPEGs cost one Hub API call each and trip its rate limit
long before the bytes matter, so each task's ``images/`` ships as a single
``images.tar`` next to its ``annotations.json``. The tar is uncompressed -- JPEGs
do not shrink, and Xet then deduplicates an edited archive chunk by chunk -- and
written deterministically, so repacking unchanged frames reproduces the same bytes
and nothing is uploaded or downloaded again.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ANNOTATIONS = "annotations.json"  # as ``mtmb.dataset``; not imported, to stay leaf
IMAGES = "images"
IMAGES_ARCHIVE = "images.tar"
#: Left in ``images/`` by ``unpack_images``: which archive the frames came from.
UNPACKED_STAMP = ".unpacked"


def _normalised(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip what varies between machines, so equal frames give equal archives."""
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info


def pack_images(task_dir: Path, out_path: Path) -> Path:
    """Write the frames ``task_dir``'s annotations list as ``out_path``.

    The annotations decide what ships, not the directory: a frame dropped from
    ``annotations.json`` but left on disk stays out, and one listed but missing
    stops the pack. Members are named as the annotations name them, ``images/<frame>``.
    """
    coco = json.loads((task_dir / ANNOTATIONS).read_text())
    names = sorted({image["file_name"] for image in coco["images"]})
    if not names:
        raise FileNotFoundError(f"no images listed in {task_dir / ANNOTATIONS}")
    if missing := [name for name in names if not (task_dir / name).is_file()]:
        raise FileNotFoundError(f"{len(missing)} listed frames missing, e.g. {missing[0]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w", format=tarfile.GNU_FORMAT) as tar:
        for name in names:
            tar.add(task_dir / name, arcname=name, filter=_normalised)
    return out_path


def _stamp(archive: Path) -> str:
    stat = archive.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def unpack_images(task_dir: Path) -> bool:
    """Extract ``task_dir/images.tar`` into ``task_dir/images/``; False if current."""
    archive = task_dir / IMAGES_ARCHIVE
    images_dir = task_dir / IMAGES
    stamp = images_dir / UNPACKED_STAMP
    if stamp.exists() and stamp.read_text() == _stamp(archive):
        return False

    with tempfile.TemporaryDirectory(dir=task_dir, prefix=".unpack-") as tmp:
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        unpacked = Path(tmp) / IMAGES
        (unpacked / UNPACKED_STAMP).write_text(_stamp(archive))
        if images_dir.exists():
            shutil.rmtree(images_dir)
        unpacked.rename(images_dir)
    return True
