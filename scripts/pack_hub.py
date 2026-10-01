"""Pack each task as the Hub ships it, and optionally push it there.

    uv run python scripts/pack_hub.py            # stage under build/hub/, upload nothing
    uv run python scripts/pack_hub.py --upload   # stage, then commit it to the Hub

Each task becomes ``<task>/annotations.json`` plus one ``<task>/images.tar`` --
the layout ``MouseDataset.download`` fetches. The upload is one commit that adds
the archives and deletes the loose ``<task>/images/*`` frames they replace; the
frames stay reachable at every earlier revision.
"""

import argparse
import shutil

from mtmb.dataset import ANNOTATIONS, HF_REPO_ID, MouseDataset
from mtmb.hub import IMAGES, IMAGES_ARCHIVE, pack_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"commit the staged tasks to {HF_REPO_ID}, removing the loose frames",
    )
    args = parser.parse_args()

    dataset = MouseDataset()
    stage_dir = dataset.build_dir / "hub"
    print(f"{dataset.path} -- {len(dataset.tasks)} tasks -> {stage_dir}")

    for task in dataset.tasks:
        task_dir = dataset.task_dir(task)
        out_dir = stage_dir / task.value
        archive = pack_images(task_dir, out_dir / IMAGES_ARCHIVE)
        shutil.copy2(dataset.annotations_path(task), out_dir / ANNOTATIONS)
        print(f"  {task.value:<20} {archive.stat().st_size / 1e6:>8.1f} MB")

    if not args.upload:
        print(f"\nstaged only; rerun with --upload to commit to {HF_REPO_ID}")
        return

    from huggingface_hub import HfApi

    commit = HfApi().upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        folder_path=stage_dir,
        commit_message="Pack each task's frames into one images.tar",
        # Only the loose frames: ``*/images.tar`` has no slash after ``images``.
        delete_patterns=[f"*/{IMAGES}/*"],
    )
    print(f"\ncommitted -> {commit.commit_url}")


if __name__ == "__main__":
    main()
