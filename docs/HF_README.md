---
pretty_name: Multi-Task Mouse Behaviour Dataset
license: cc-by-nc-4.0
task_categories:
  - object-detection
  - image-segmentation
  - keypoint-detection
tags:
  - animal-pose
  - behavioral-neuroscience
  - mouse
  - instance-segmentation
  - multi-object-tracking
  - video
  - sam3
  - deeplabcut
size_categories:
  - 10K<n<100K
---

<table>
<tr>
<td align="center"><img src="assets/barnes_maze.gif" width="260"><br><code>barnes_maze</code></td>
<td align="center"><img src="assets/direct_interaction.gif" width="260"><br><code>direct_interaction</code></td>
<td align="center"><img src="assets/marble.gif" width="260"><br><code>marble</code></td>
</tr>
<tr>
<td align="center"><img src="assets/nort.gif" width="260"><br><code>nort</code></td>
<td align="center"><img src="assets/nort2.gif" width="260"><br><code>nort2</code></td>
<td align="center"><img src="assets/open_field.gif" width="260"><br><code>open_field</code></td>
</tr>
<tr>
<td align="center"><img src="assets/social_interaction.gif" width="260"><br><code>social_interaction</code></td>
<td align="center"><img src="assets/t_maze.gif" width="260"><br><code>t_maze</code></td>
<td align="center"><img src="assets/three_chamber.gif" width="260"><br><code>three_chamber</code></td>
</tr>
</table>

Six-second clips, annotations overlaid (boxes, masks, keypoints, track ids).

# Multi-Task Mouse Behaviour Dataset

Bounding boxes, instance masks, 27-point poses and persistent track ids for
laboratory mice, **on the same frames**, across nine standard behavioural
assays. 14,300 frames, 27,716 annotated instances, 1280×720.

Most animal-behaviour datasets give you one annotation type. Training or
evaluating a model that does detection *and* segmentation *and* pose *and*
tracking usually means stitching together sources that disagree on species,
viewpoint, and label conventions. This dataset provides all four on every frame,
under one category schema, from real recordings of running experiments.

## Where the annotations come from

Every label is **machine-generated** by the DeepLabSAM pipeline.
The pipeline is a two-stage composition:

1. **Detection, segmentation and tracking — SAM 3 (video).** A text prompt
   (`"mouse"`) drives promptable detection and tracking, producing a pixel-accurate
   mask and a persistent `track_id` per animal per frame.
2. **Pose — DeepLabCut SuperAnimal (`superanimal_topviewmouse`), mask-gated.**
   Each instance is cropped to its own box and the crop's background is zeroed
   **using that instance's mask** before it reaches the pose head. Masking is
   applied after normalisation, where the network's neutral value is 0, so the
   background drops to that baseline instead of becoming an out-of-distribution
   black box.

Mask-gating is what makes stage 2 usable as an annotator. A top-down pose model fed
a raw crop of two touching animals has no way to know which one it should fit;
gated by the mask, the intended animal is the only signal present. That is why the
poses survive the crowded assays (`open_field`, `social_interaction`) where a plain
detect-then-crop cascade puts keypoints on the neighbour.

## Tasks

Each task is one continuous recording of a standard behavioural assay, filmed from
above and decoded to consecutive frames at 30 fps. Frames are contiguous, so
**neighbours are near-duplicates** — the ~14k frames are roughly 7.9 minutes of
footage, not 14k independent samples.

| Task | Frames | Instances | Animals | ≈ Duration | Median box (px²) |
|---|---:|---:|---:|---:|---:|
| `barnes_maze` | 1,847 | 1,847 | 1 | 62 s | 8,241 |
| `direct_interaction` | 1,430 | 2,860 | 2 | 48 s | 92,987 |
| `marble` | 1,797 | 1,797 | 1 | 60 s | 102,820 |
| `nort` | 1,719 | 1,708 | 1 | 57 s | 2,880 |
| `nort2` | 1,053 | 1,053 | 1 | 35 s | 22,518 |
| `open_field` | 1,939 | 7,690 | 4 | 65 s | 1,377 |
| `social_interaction` | 1,472 | 5,888 | 4 | 49 s | 2,352 |
| `t_maze` | 2,128 | 2,128 | 1 | 71 s | 1,409 |
| `three_chamber` | 915 | 2,745 | 3 | 30 s | 2,816 |
| **Total** | **14,300** | **27,716** | | **≈ 7.9 min** | |

**`barnes_maze`** — Spatial-learning assay. One mouse on a circular platform ringed
with escape holes.

**`direct_interaction`** — Two mice in a small bedding-filled arena with no divider
between them. The only task with two coat colours.

**`marble`** — Marble-burying assay for repetitive and anxiety-like behaviour.

**`nort` / `nort2`** — Novel Object Recognition: one mouse investigating objects in
an open arena, recorded twice in different arenas. The only within-assay pair.

**`open_field`** — Locomotion and anxiety assay. Four mice in a quadrant-divided
arena, one per compartment.

**`social_interaction`** — Four mice in a divided arena, approaching and contacting
one another across dividers.

**`t_maze`** — Spatial working-memory assay. One mouse running the arms of a
T-shaped maze.

**`three_chamber`** — Sociability assay. Three mice in a chamber divided by
transparent partitions.

Four of nine tasks are multi-animal.

## Layout

```
dataset/<task>/
  images/frame_00000.jpg ...        # consecutive decoded frames, 1280x720
  annotations.json                  # COCO: bbox, segmentation, track_id, keypoints
  task.yaml                         # per-task metadata (see below)
  viz/frame_00000.png               # per-frame overlays (derived)
splits/<name>.json                  # split manifests -- the reproducible artifact
build/                              # materialised exports (derived, git-ignored)
udmt/                               # registry, merge, splits, exports, validation
```

Per-task files are canonical; anything under `build/` is regenerable.

### Annotation schema

Standard COCO detection + keypoints, one category (`mouse`, id 1), with a few
additions:

| Field | Meaning |
|---|---|
| `track_id` | Persistent identity across frames within a task. |
| `score` | SAM 3's detection confidence. |
| `keypoints` | Flat `[x, y, v] × 27`, SuperAnimal `topviewmouse` order. |
| `keypoint_scores` | Per-keypoint model confidence — use this to re-threshold. |
| `num_keypoints` | How many cleared the 0.3 threshold. |
| `images[*].frame_index` | Position in the source recording (temporal order). |

The 27 keypoints follow SuperAnimal `topviewmouse` naming exactly, so DeepLabCut
models are directly comparable:

`nose`, `left_ear`, `right_ear`, `left_ear_tip`, `right_ear_tip`, `left_eye`,
`right_eye`, `neck`, `mid_back`, `mouse_center`, `mid_backend`, `mid_backend2`,
`mid_backend3`, `tail_base`, `tail1`, `tail2`, `tail3`, `tail4`, `tail5`,
`left_shoulder`, `left_midside`, `left_hip`, `right_shoulder`, `right_midside`,
`right_hip`, `tail_end`, `head_midpoint`

## License

CC BY-NC 4.0. The pose annotations are the output of DeepLabCut's
`superanimal_topviewmouse` checkpoint, whose weights (distinct from the LGPL-3.0
DeepLabCut codebase) are licensed for academic, non-commercial use only. That
restriction is what this dataset inherits and passes on; it is not a choice made
independently of the pipeline that produced the labels.

## Acknowledgments

Annotations were produced with:

- **SAM 3** (Meta AI / FAIR) — detection, segmentation and tracking.
- **DeepLabCut SuperAnimal, `superanimal_topviewmouse`** (Mathis Lab, EPFL) — pose
  estimation. See [Ye et al., 2024](https://www.nature.com/articles/s41467-024-48792-2).

Recordings were collected at [IGF, CNRS](https://www.igf.cnrs.fr/) and [UPV/EHU]().

## Citation

A paper describing this dataset is in preparation. Citation details (BibTeX) will
be added here on publication.

```bibtex
@dataset{mtmb,
  title   = {Multi-Task Mouse Behaviour Dataset},
  author  = {TBD},
  year    = {TBD},
  note    = {TBD},
}
```
