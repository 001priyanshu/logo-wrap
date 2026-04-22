# Logo Wrap

Wrap a logo around the subject of a photograph so that part of the logo sits
**in front of** the subject and part sits **behind** — the same "embracing"
composition you see in polished editorial ads, but generated automatically
from a single input image and a logo PNG.

| Input | Output |
| :---: | :---: |
| ![Input](./assets/01_input.jpg) | ![Output](./assets/05_output.jpg) |

---

## Pipeline at a glance

```
 Input image ──► BiRefNet segmentation ──► Mask cleanup ──► Adaptive
                                                            placement
                                                                │
                    ┌───────────────────────────────────────────┘
                    ▼
       Front / behind partition ──► Alpha composite ──► Output
```

Four visible stages, each shown below.

### Stage 1 — Subject segmentation (BiRefNet)

The raw image is sent to BiRefNet (hosted on fal.ai). It returns a
high-resolution binary mask of the subject. This mask is what lets us decide
which pixels of the logo should go in front of the person and which should
go behind.

![Raw mask](./assets/02_mask_raw.png)

### Stage 2 — Mask cleanup

Neural masks are rarely clean — they have small holes, speckle, and sometimes
spurious blobs (a hand's reflection, a detached shadow). Two operations fix
this:

1. **Morphological closing** with a 7×7 kernel fills small holes.
2. **Connected-component filtering** keeps the largest blob, plus any blob
   whose area is ≥ 25 % of the largest (so that e.g. a raised arm or a second
   person is not discarded).

```
cleaned = close(mask, 7×7)
keep {c ∈ components(cleaned) : area(c) ≥ 0.25 · max_area}
```

![Cleaned mask](./assets/03_mask_clean.png)

### Stage 3 — Adaptive placement

Given the cleaned mask we compute two quantities that decide *where* and
*how big* the logo should be.

**Width clamp.** Let the subject's bounding box have width `s_w` and sit at
x-range `[x_min, x_max]` in an image of width `W`. Let the user request a
width factor `w` (logo width as a multiple of subject width). Define the
normalized free space on each side:

$$
\ell = \frac{x_{\min}}{s_w}, \qquad r = \frac{W - x_{\max}}{s_w}
$$

and clamp the width factor so the logo never overruns the frame:

$$
w_{\text{adj}} = \min\bigl(w,\; \ell + 1.15,\; r + 1.15\bigr)
$$

The constant `1.15` comes from the fact that the logo extends `w/2` on each
side of the subject's center, and we allow a small overshoot of 0.15·s_w into
the background before clipping — this was tuned on ~60 reference images.

**Vertical offset.** The "wrap line" (the horizontal level where the logo
crosses the subject) depends on two things: how tall-and-skinny the subject
is, and how big the logo itself is. We measured the visually-correct wrap
line across a calibration set of images and fit a linear model over the two
predictors:

$$
\text{aspect} = \frac{s_h}{s_w}
$$

$$
v = 0.905 \;-\; 0.133 \cdot \text{aspect} \;-\; 0.0623 \cdot w_{\text{adj}}
$$

Intuition:
- **Taller subjects** (larger aspect ratio) ⇒ wrap line moves *up* on the body.
- **Bigger logos** (larger `w_adj`) ⇒ wrap line also moves *up*, because a
  larger logo's center already reaches the torso.

The fit was obtained by least-squares on manually-placed samples; on the
hold-out set the residual standard deviation on `v` was ~0.03, well within
the tolerance where the result still looks natural.

![Placement diagram](./assets/04_placement.png)

### Stage 4 — Front / behind partition

A single logo PNG has to be split into "this part goes in front of the
subject" and "this part goes behind". We classify each logo pixel using a
small geometric rule:

1. A **rotated rectangle** of size `(0.20·wL, 0.45·hL)` at angle 40° is drawn
   through the logo's center. Pixels inside this rectangle are **front**.
2. Pixels in the **top-right** and **bottom-left** quadrants that fall
   *outside* the rectangle are also **front** (this is what gives the
   diagonal wrap look).
3. Everything else is **behind**.

Formally, after translating coordinates to the logo center and rotating by
θ = 40°:

$$
(X_r, Y_r) = (X_c \cos\theta + Y_c \sin\theta,\; -X_c \sin\theta + Y_c \cos\theta)
$$

$$
\text{front} = \Bigl(|X_r| \le \tfrac{rw}{2} \wedge |Y_r| \le \tfrac{rh}{2}\Bigr) \;\cup\; \bigl(Q_{TR} \cup Q_{BL}\bigr) \setminus \text{rect}
$$

$$
\text{behind} = \text{logo} \setminus \text{front}
$$

The behind layer is additionally multiplied by `(1 - mask_eroded)` so that it
is erased wherever the subject is — that's what creates the occlusion. Final
composite: `base → behind layer → front layer`.

---

## Usage

```bash
pip install -r requirements.txt
export FAL_KEY=your_fal_api_key

python logo_wrap_standalone.py \
  --input "https://example.com/photo.jpg" \
  --logo ./logo.png \
  --output-dir ./out \
  --width-factor 1.4
```

Outputs land in `./out/`:
- `input_image.*` — downloaded original
- `masked_image.png` — BiRefNet mask
- `output_image.jpeg` — final composite
- `json_params.json` — parameters actually used (useful for reproducing a run)

### Useful flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--width-factor` | 1.4 | Logo width as a multiple of subject width |
| `--horizontal-offset` | 0.5 | 0 = left edge of subject, 1 = right edge |
| `--vertical-offset` | 0.55 | Only used with `--strict`; otherwise fitted |
| `--strict` | false | Skip the adaptive fit, use the values as-is |
| `--rect-angle-deg` | 40 | Angle of the front-rectangle partition |
| `--logo-opacity` | 0.9 | Opacity applied to the logo before compositing |
| `--mask-image-url` | — | Skip BiRefNet and use this mask URL instead |

---

## Why this is interesting

- **No diffusion model.** The entire effect is classical compositing driven
  by one segmentation call. It runs in ~1–2 s after segmentation, on CPU.
- **Fully parametric.** Every placement decision reduces to two scalars
  (`w_adj`, `v`), both with interpretable formulas. That means failures can
  be debugged by inspecting the mask and the two numbers, rather than
  re-prompting a generative model.
- **Extensible.** Swapping the rotated-rectangle rule for a learned
  partition, or conditioning `v` on pose keypoints, are both drop-in
  changes.

---

## File layout

```
logo-wrap-standalone/
├── logo_wrap_standalone.py   # single-file pipeline
├── requirements.txt
├── README.md
└── assets/                   # images referenced above
    ├── 01_input.jpg
    ├── 02_mask_raw.png
    ├── 03_mask_clean.png
    ├── 04_placement.png
    └── 05_output.jpg
```
