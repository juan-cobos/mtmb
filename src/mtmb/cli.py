"""Command line: build a split of the dataset into ``build/``."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from mtmb.dataset import DEFAULT_PATH, MouseDataset, Task
from mtmb.manifest import Manifest
from mtmb.merge import LinkMode
from mtmb.splits import ValidMode

app = typer.Typer(
    add_completion=False,
    help="Build exports from the multi-task mouse dataset.",
    no_args_is_help=True,
)

TasksOpt = Annotated[
    list[Task] | None,
    typer.Option("--task", "-t", help="Task to include; repeat. Default: all of them."),
]
NameOpt = Annotated[str, typer.Option("--name", "-n", help="Export name under build/.")]
SeedOpt = Annotated[int, typer.Option("--seed", help="Seed for the random draw.")]
EveryOpt = Annotated[
    int,
    typer.Option("--every", "-e", min=1, help="Keep every Nth frame (30 fps source)."),
]
LinkOpt = Annotated[
    LinkMode,
    typer.Option("--link", help="How images are placed beside annotations."),
]
OverwriteOpt = Annotated[
    bool,
    typer.Option("--overwrite", help="Rebuild an export that already exists."),
]
ThresholdOpt = Annotated[
    float,
    typer.Option(
        "--keypoint-threshold",
        min=0.0,
        max=1.0,
        help="Score at or above which a keypoint is labelled visible.",
    ),
]
DatasetOpt = Annotated[
    Path | None,
    typer.Option("--dataset", help="Dataset directory to read."),
]
BuildOpt = Annotated[
    Path | None,
    typer.Option("--build", help="Where exports are written."),
]


@contextmanager
def _reported() -> Generator[None]:
    """Report a bad split or a failed build as a message, not a traceback."""
    try:
        yield
    except (RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
        typer.secho(f"{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


def _build(
    dataset: MouseDataset,
    manifest: Manifest,
    link: LinkMode,
    threshold: float,
    overwrite: bool = False,
) -> None:
    typer.echo(manifest.describe())
    dataset.build(
        manifest,
        link=link,
        keypoint_threshold=threshold,
        overwrite=overwrite,
    )


@app.command()
def pooled(
    tasks: TasksOpt = None,
    ratios: Annotated[
        tuple[float, float, float],
        typer.Option("--ratios", help="Train/valid/test fractions; must sum to 1."),
    ] = (0.7, 0.15, 0.15),
    name: NameOpt = "pooled",
    seed: SeedOpt = 0,
    every: EveryOpt = 1,
    link: LinkOpt = LinkMode.symlink,
    keypoint_threshold: ThresholdOpt = 0.3,
    overwrite: OverwriteOpt = False,
    dataset_path: DatasetOpt = None,
    build_dir: BuildOpt = None,
) -> None:
    """Pool the tasks and split at the image level."""
    dataset = MouseDataset(dataset_path or DEFAULT_PATH, build_root=build_dir)
    with _reported():
        manifest = dataset.split_random(tasks, ratios, name, seed, every)
        _build(dataset, manifest, link, keypoint_threshold, overwrite)


@app.command("by-task")
def by_task(
    train: Annotated[
        list[Task],
        typer.Option("--train", help="Task to train on; repeat."),
    ],
    valid: Annotated[
        list[Task],
        typer.Option("--valid", help="Task to validate on; repeat."),
    ],
    test: Annotated[
        list[Task] | None,
        typer.Option("--test", help="Task to test on; repeat."),
    ] = None,
    name: NameOpt = "by_task",
    every: EveryOpt = 1,
    link: LinkOpt = LinkMode.symlink,
    keypoint_threshold: ThresholdOpt = 0.3,
    overwrite: OverwriteOpt = False,
    dataset_path: DatasetOpt = None,
    build_dir: BuildOpt = None,
) -> None:
    """Hold out whole tasks -- the protocol with no frame-level leakage."""
    dataset = MouseDataset(dataset_path or DEFAULT_PATH, build_root=build_dir)
    with _reported():
        manifest = dataset.split_by_task(train, valid, list(test or []), name, every)
        _build(dataset, manifest, link, keypoint_threshold, overwrite)


@app.command()
def loo(
    tasks: TasksOpt = None,
    name: NameOpt = "loao",
    valid_mode: Annotated[
        ValidMode,
        typer.Option("--valid-mode", help="Valid from train, or a task of its own."),
    ] = ValidMode.in_domain,
    valid_fraction: Annotated[
        float,
        typer.Option(
            "--valid-fraction",
            min=0.0,
            max=1.0,
            help="Valid slice of train; in_domain only.",
        ),
    ] = 0.1,
    folds: Annotated[
        int | None,
        typer.Option(
            "--folds",
            "-f",
            min=1,
            help="Build only N folds, held-out tasks drawn at random. Default: all.",
        ),
    ] = None,
    seed: SeedOpt = 0,
    every: EveryOpt = 1,
    link: LinkOpt = LinkMode.symlink,
    keypoint_threshold: ThresholdOpt = 0.3,
    overwrite: OverwriteOpt = False,
    dataset_path: DatasetOpt = None,
    build_dir: BuildOpt = None,
) -> None:
    """Build one export per task, that task held out as test."""
    dataset = MouseDataset(dataset_path or DEFAULT_PATH, build_root=build_dir)
    with _reported():
        manifests = dataset.split_leave_one_out(
            tasks,
            name,
            valid_fraction,
            seed,
            every,
            folds,
            valid_mode,
        )
        for fold, manifest in enumerate(manifests, 1):
            typer.echo(f"[{fold}/{len(manifests)}] ", nl=False)
            _build(dataset, manifest, link, keypoint_threshold, overwrite)


@app.command()
def tasks(
    dataset_path: DatasetOpt = None,
    build_dir: BuildOpt = None,
) -> None:
    """List the tasks and what has already been built."""
    dataset = MouseDataset(dataset_path or DEFAULT_PATH, build_root=build_dir)
    typer.echo(f"{dataset.path} -- {len(dataset.tasks)} tasks")
    for row in dataset.summary():
        typer.echo(
            f"  {row['index']} {row['task']:<20} {row['images']:>6} images "
            f"{row['annotations']:>6} annotations  {row['max_instances']} per frame",
        )
    typer.echo(
        f"\n{dataset.build_dir} -- {', '.join(dataset.manifests()) or 'nothing'}",
    )


@app.command()
def download(
    dataset_path: DatasetOpt = None,
) -> None:
    """Fetch the dataset. Not yet available."""
    dataset = MouseDataset(dataset_path or DEFAULT_PATH)
    dataset.download()


if __name__ == "__main__":
    app()
