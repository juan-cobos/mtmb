"""Splitting, exporting and validating the Multi-Task Mouse Behaviour Dataset.s"""

from importlib.metadata import version

from mtmb.dataset import ALL_TASKS, HF_REPO_ID, MouseDataset, Task
from mtmb.manifest import Manifest, Split

__version__ = version("mtmb")

__all__ = [
    "ALL_TASKS",
    "HF_REPO_ID",
    "Manifest",
    "MouseDataset",
    "Split",
    "Task",
    "__version__",
]
