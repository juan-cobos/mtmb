"""Load the dataset and export one annotated video per task."""

from mtmb.dataset import MouseDataset

if __name__ == "__main__":
    dataset = MouseDataset()
    print(f"{dataset.path} -- {len(dataset.tasks)} tasks")
    for path in dataset.annotate_videos():
        print(f"  -> {path}")
