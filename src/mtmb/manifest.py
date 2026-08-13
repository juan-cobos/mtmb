"""What a split *is*: the tasks and global image ids in each of train/valid/test."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mtmb.dataset import ANNOTATIONS, SPLITS, MouseDataset

MANIFEST = "manifest.json"


@dataclass(frozen=True)
class Split:
    """One split's contents: task names and the global image ids drawn from them."""

    tasks: list[str]
    image_ids: list[int]

    def __len__(self) -> int:
        return len(self.image_ids)

    def to_dict(self) -> dict:
        return {"tasks": list(self.tasks), "image_ids": list(self.image_ids)}

    @classmethod
    def from_dict(cls, data: dict) -> Split:
        return cls(tasks=list(data["tasks"]), image_ids=list(data["image_ids"]))


@dataclass(frozen=True)
class Manifest:
    """A saved split: which tasks, which images, and what produced them.

    ``task_index``, ``annotations`` and ``every`` are provenance -- enough to tell
    whether the manifest still describes the dataset in front of you, and enough to
    reproduce it. ``sources`` records each task's full image count, so a subsampled
    split still shows what fraction of the footage it drew from.
    """

    name: str
    mode: str
    splits: dict[str, Split]
    created: str = ""
    seed: int | None = None
    annotations: str = ANNOTATIONS
    every: int = 1
    task_index: dict[str, int] = field(default_factory=dict)
    sources: dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------- construction

    @classmethod
    def make(
        cls,
        dataset: MouseDataset,
        name: str,
        mode: str,
        splits: dict[str, Split],
        seed: int | None = None,
        every: int = 1,
    ) -> Manifest:
        """Assemble a manifest, checking that no image landed in two splits."""
        unknown = [n for n in splits if n not in SPLITS]
        if unknown:
            raise ValueError(
                f"unknown split name(s) {unknown}; expected {list(SPLITS)}",
            )

        seen: dict[int, str] = {}
        for split_name, split in splits.items():
            for image_id in split.image_ids:
                if image_id in seen:
                    raise ValueError(
                        f"image {image_id} is in both {seen[image_id]} and {split_name}",
                    )
                seen[image_id] = split_name

        tasks = sorted({t for s in splits.values() for t in s.tasks})
        return cls(
            name=name,
            mode=mode,
            # An empty split is dropped, not recorded: `build` cannot merge zero
            # tasks, and RF-DETR treats a missing test directory as fine.
            splits={
                n: Split(splits[n].tasks, sorted(splits[n].image_ids))
                for n in SPLITS
                if n in splits and splits[n].image_ids
            },
            created=datetime.now(UTC).isoformat(timespec="seconds"),
            seed=seed,
            annotations=ANNOTATIONS,
            every=every,
            task_index={t: dataset.task(t).get_index() for t in tasks},
            sources={t: {"images": len(dataset.image_ids(t))} for t in tasks},
        )

    # ------------------------------------------------------------------- fields

    @property
    def tasks(self) -> list[str]:
        """Every task this manifest touches, in name order."""
        return sorted({t for s in self.splits.values() for t in s.tasks})

    def __len__(self) -> int:
        return sum(len(s) for s in self.splits.values())

    def describe(self) -> str:
        parts = [
            f"{n} {len(s)} imgs ({', '.join(s.tasks)})"
            for n, s in self.splits.items()
            if s.image_ids
        ]
        stride = f" every={self.every}" if self.every > 1 else ""
        return f"{self.name} [{self.mode}{stride}] " + " | ".join(parts)

    def matches(self, other: Manifest) -> bool:
        """Same split as ``other``, ignoring when each was created."""
        return {k: v for k, v in self.to_dict().items() if k != "created"} == {
            k: v for k, v in other.to_dict().items() if k != "created"
        }

    # ------------------------------------------------------------- consistency

    def verify(self, dataset: MouseDataset) -> None:
        """Raise if the id space this manifest was written against has moved.

        A manifest stores global ids, which are only meaningful under the task index
        that produced them; ``Task``'s declaration order is that index. Reordering it
        renumbers tasks and turns every stored id into a pointer at the wrong frames.
        """
        stale = {
            t: (i, dataset.task(t).get_index())
            for t, i in self.task_index.items()
            if dataset.task(t).get_index() != i
        }
        if stale:
            moved = ", ".join(f"{t}: {was} -> {now}" for t, (was, now) in stale.items())
            raise ValueError(
                f"{self.name!r} was saved under a different task index ({moved}); "
                "its global image ids no longer point at the frames they did -- "
                "rebuild the split",
            )
        if self.annotations != ANNOTATIONS:
            raise ValueError(
                f"{self.name!r} indexed {self.annotations!r}, but this dataset reads "
                f"{ANNOTATIONS!r}",
            )

    # ---------------------------------------------------------------------- i/o

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "created": self.created,
            "seed": self.seed,
            "annotations": self.annotations,
            "every": self.every,
            "task_index": self.task_index,
            "sources": self.sources,
            "splits": {n: s.to_dict() for n, s in self.splits.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        return cls(
            name=data["name"],
            mode=data["mode"],
            # Ordered as SPLITS regardless of how the file happens to be written.
            splits={
                n: Split.from_dict(data["splits"][n]) for n in SPLITS if n in data["splits"]
            },
            created=data.get("created", ""),
            seed=data.get("seed"),
            annotations=data.get("annotations", ANNOTATIONS),
            every=data.get("every", 1),
            task_index=data.get("task_index", {}),
            sources=data.get("sources", {}),
        )

    def save(self, out_dir: Path) -> Path:
        """Write to ``<out_dir>/manifest.json``, beside the splits it produced."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / MANIFEST
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, dataset: MouseDataset, name: str) -> Manifest:
        """Read the manifest out of an export, refusing one the dataset has moved on from.

        ``name`` is an export under ``build_dir``; pass a path to read one that was
        built somewhere else.
        """
        out_dir = Path(name) if Path(name).is_absolute() else dataset.build_dir / name
        path = out_dir / MANIFEST
        if not path.exists():
            known = ", ".join(dataset.manifests()) or "none"
            raise FileNotFoundError(f"no manifest at {path}; built splits: {known}")
        with open(path) as f:
            manifest = cls.from_dict(json.load(f))
        manifest.verify(dataset)
        return manifest
