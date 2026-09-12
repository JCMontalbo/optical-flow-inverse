# Downstream experiment 2: cardiac MRI, where blind augmentation is not free

*Pre-registered. Written after the DAVIS result ([downstream_plan.md](downstream_plan.md)) and before any
training run on this data.*

## Why a second experiment

DAVIS answered a question the dissertation did not ask. On natural video, flips and large rotations produce
valid images, so blind augmentation is never penalised for being blind, and it won. The dissertation's claim
(ch. 1) is specifically about anatomy: a flipped or 30°-rotated MRI slice is not a plausible patient, so
augmentation should come from *observed* motion. Two fixes relative to DAVIS:

1. **The arena**: MSD Task02 Heart — 20 cardiac MRI volumes (320×320×~110, 1.25 mm in-plane, 1.37 mm
   between slices), left atrium labelled. Adjacent slices are the "frame pair" (dissertation Fig. 3.2);
   the recovered flow is the anatomical change through depth. Adjacent-slice mask IoU ≈ 0.88.
2. **The method sampled on the fly**, not as a fixed set: the recovered flow defines a continuous family of
   perturbations, and a fresh member is drawn every training step, exactly as the baselines draw a fresh
   transform every step.

And a baseline the dissertation never compared against: **random elastic deformation** (Ronneberger et al.,
2015) — smooth random displacement fields, the standard augmentation for medical segmentation. It is also
geometry-free and also produces smooth deformations. The sharp question is therefore:

> Do deformations derived from observed anatomical motion train a better segmenter than random smooth
> deformations of the same magnitude?

## Data

- 20 volumes, split by patient: 14 train, 6 val (every third volume in sorted id order goes to val).
- Slices along z; 192×192 crop at rows 48:240, cols 72:264 (covers the atrium's extent in every volume);
  intensity clipped at the volume's 99.5th percentile and scaled to [0, 1].
- Per volume, the slices in the labelled z-range ± 3 form the pool (≈ 70 slices).
- **Label budget** — labelled slices per training volume: **1, 2, 4, all** (≈ 14, 28, 56, ~950 slices),
  evenly spaced within the labelled range, same choice for every arm and seed.
- **Metric** — 3D Dice of the left atrium per val volume (2D predictions stacked over the pool),
  mean over the 6 val volumes, mean ± std over 3 seeds.

## Arms

All augmented arms share the same intensity jitter (gamma 0.8–1.25, contrast 0.9–1.1, additive noise σ 0.02),
so the arms differ only in geometry. **No flips anywhere.**

| arm | geometry each step | what it tests |
|---|---|---|
| A `none` | none | floor |
| B `plausible` | rotation ±10°, scale 0.9–1.1, translation ±6 px | the strongest *anatomically plausible* affine augmentation |
| C `elastic` | random smooth displacement: Gaussian noise smoothed with σ = 12 px, scaled to a random peak of 0–8 px | the standard medical augmentation; geometry-free deformation |
| D `flow` | a fresh member of the recovered-flow family: neighbour j ∈ {k±1, k±2} (PSNR-gated), ε ~ U(0.25, 1.75); with p = ½ a Gaussian window (centre near the atrium, width 12–32 px) gating the motion with gain U(0, 3); with p = ½ a windowed area-preserving generator (rotation / squeeze / shear / saddle, peak 0–8 px); image and label warped together | the dissertation's method, sampled |
| E `propagate` | labels carried to the accepted real neighbours k±1, k±2 (fixed pairs) | is it the perturbation or the neighbours? |

Deformation magnitudes for C and D are matched: both peak at ≤ 8 px plus the observed flow (mean ≈ 1–2 px).

## Model and training

Same U-Net as DAVIS (1 input channel, 1.9 M parameters), BCE + Dice, Adam 1e-3 cosine, batch 16,
**3,000 steps for every cell**, early stopping on 5 held-out labelled slices per *training* volume (never
val), checkpoint at the best. 4 budgets × 5 arms × 3 seeds = 60 cells, ≈ 1.5 min each, three in parallel.

## Pre-registered criteria

- **H1 (the dissertation's claim, in its own arena)**: at 1 and 2 labelled slices/volume, D `flow` beats
  B `plausible` by ≥ 2 Dice points with disjoint seed ranges.
- **H2 (the sharp question)**: D `flow` beats C `elastic` by ≥ 2 points at those budgets. If C ≈ D, the
  conclusion is "smooth deformation helps; its provenance does not", and that is reported as such.
- **H3**: D beats E `propagate` by ≥ 1 point.
- **H4** (descriptive): the D − B and D − C gaps at `all` labels.
- Any outcome is reported against these lines. Nothing is tuned per arm after seeing val.

## Risks

| risk | handling |
|---|---|
| Horn–Schunck flow poor on MRI (low texture inside the blood pool) | PSNR gate at 25 dB as before; report the rejection rate; the atrium interior is smooth, so the flow there is regularisation-driven — part of the result |
| Elastic and flow magnitudes not comparable | both capped at 8 px peak; report the mean displacement actually applied per arm |
| 6 val volumes is few | per-volume Dice reported; seeds are the replication |
| Atrium is tiny (0.3–0.5 % of voxels) | Dice, not IoU on the whole slice; crop keeps it central |
