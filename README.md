# Practice in Text, Act in Pixels: Correcting Where Perception Fails

Code for the paper of the same title: [arXiv:2611.22722](https://arxiv.org/abs/2611.22722).

GUI agents can practise cheaply on a text version of an interface and act on real screenshots through a
vision-language *screen reader* that turns each screenshot into the same kind of list. Success drops sharply from text
to screenshots, because the reader misreads elements and lists them in its own order, and its misreads recur on the
same elements. *Reader alignment* fine-tunes the reader on about 2,000 screenshots so that its output for a screenshot
is that screen's text view, with the same content and order as in practice. The policy is never trained on pixels.

On MobileGym, with the same budget of about 2,000 screenshots, a policy that practised only in text and acts through
the aligned reader completes 80.9% of the confirmation tasks, against 55.6% for a policy trained directly on
screenshots through the original reader.

## What is here

| Directory | Contents |
|---|---|
| `textview/` | The text view of a MobileGym page: a visible-element extractor run in the browser (`extract.js`), icon names, rendering for the policy |
| `reader/` | Reader calls and the line parser; matching a reader's list to the true text view; the element error rate and the acceptance test of an aligned reader; recurrence of misreads; alignment labels, LoRA training and merging; re-reading saved screenshots |
| `practice/` | Corrupted practice environments: generic noise, and noise matched to the reader's error rates |
| `policy/` | The episode runner (text view, real pipeline, noise, program oracles); the policy prompt; the 29 program oracles; collecting judge-accepted episodes; LoRA training; picking episodes by screenshot budget |
| `evaluation/` | The paired bootstrap over templates and the fixed wording rules |
| `androidcontrol/` | Optional: the text view of AndroidControl screens and the two reader-side measurements on it (scripts only, no data) |
| `configs/` | Model revisions, reader and policy settings, LoRA hyperparameters, templates and instance seeds, path variables |
| `scripts/` | MobileGym setup, model download, model servers, the step-by-step reproduction, a smoke test |
| `tests/` | Small CPU tests |

Not included: model weights, MobileGym itself and anything generated from it (screenshots, text views, labels),
AndroidControl's data, the
scripts that made the paper's tables and figures, and some secondary analyses (the element-replay simulator and the
gap decomposition, the look-up reader, the same-label adaptation baseline).

## Requirements

- Linux with one NVIDIA GPU of 48 GB (the study used RTX A6000s) and a driver for CUDA 12.8. One GPU holds the policy
  and the reader side by side.
- Python 3.11, Node.js 22, git.
- Disk: about 26 GB for the two base models, 2 GB for MobileGym and its browser, and room for runs (screenshots).

## Installation

```bash
git clone https://github.com/Carlxuzhl/practice-in-text.git && cd practice-in-text
cp configs/paths.env.example configs/paths.env    # edit the paths
source configs/paths.env
```

Two environments, as in the study: one for the simulator, the analyses and the tests (Python and Node.js), one for the
GPU work (Python only). With conda:

```bash
conda create -y -p ~/envs/ptap-sim -c conda-forge python=3.11 'nodejs>=22,<23' pip
~/envs/ptap-sim/bin/pip install -r requirements/sim.txt
conda create -y -p ~/envs/ptap-gpu -c conda-forge python=3.11 pip
~/envs/ptap-gpu/bin/pip install -r requirements/gpu.txt
export SIM_PY=~/envs/ptap-sim/bin/python GPU_PY=~/envs/ptap-gpu/bin/python VLLM_BIN=~/envs/ptap-gpu/bin/vllm
export PATH=~/envs/ptap-sim/bin:$PATH    # node and npm for the simulator
```

Keep Node.js out of the GPU environment, and do not build it on top of the simulator environment's interpreter: with
Node.js present, conda picks an sqlite build linked to ICU, which needs a newer libstdc++ than older distributions
(such as Ubuntu 20.04) provide; torch loads the system one first and vLLM then fails with `CXXABI_1.3.15 not found`.

**MobileGym** (not included here). `scripts/setup_sim.sh` clones [MobileGym](https://github.com/Purewhiter/mobilegym)
into `$MOBILEGYM_DIR` at commit `5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38`, installs its Node packages with
`npm ci --ignore-scripts`, builds the simulator, installs Playwright's Chromium and builds the text view's icon table
from MobileGym's `lucide-react` package:

```bash
bash scripts/setup_sim.sh
```

MobileGym's code is under Apache-2.0 and its data under CC BY-NC 4.0 (academic, non-commercial use). The study ran
without MobileGym's optional dataset directory (`mobilegym-data`); do not add it. Serve the simulator while you run
episodes:

```bash
bash scripts/serve_sim.sh
```

**Models** (not included here). Download Qwen3-8B (policy) and Qwen3-VL-4B-Instruct (reader) at the revisions in
`configs/models.json` from the Hugging Face Hub; check each model's licence on its model card.

```bash
$GPU_PY scripts/download_models.py policy reader
```

`reader_8b` (Qwen3-VL-8B-Instruct) and `reader_other_family` (InternVL3-8B) are needed only for the comparisons with
other readers.

## Check the installation

```bash
$SIM_PY -m pytest -q tests          # CPU only; extractor tests need Chromium, MobileGym tests need MOBILEGYM_DIR
bash scripts/smoke.sh               # CPU: tests, a live text view, one episode with stand-in models
bash scripts/smoke.sh --gpu 0       # also: serve the reader, read 4 screenshots, one alignment step, one episode
```

## Reproducing the study

Run from the repository root with `configs/paths.env` sourced, `SIM_PY`, `GPU_PY` and `VLLM_BIN` set, `GPU` set to
the card to use, and the simulator served. Each script prints the run directories it made under `$WORK_DIR/runs/`.

| Step | Script | What it does | Screenshots spent |
|---|---|---|---|
| 1 | `scripts/reproduce/01_start_policy.sh` | Program oracles record text-view demonstrations (29 templates x seeds 1000-1007); the start policy is a LoRA on them | 0 |
| 2 | `scripts/reproduce/02_calibration_set.sh` | The start policy in the real pipeline on seeds 1008-1016; the first 2,000 reads are the calibration set (each screenshot stored with its true text view); rate-matched noise is fitted on it | 2,000 |
| 3 | `scripts/reproduce/03_practice.sh GROUP SEED` | Two practice rounds of one group (`clean`, `generic`, `rate`, `direct`, `combined`) and training seed (0, 1) | 0, or 1,000 for `direct` and `combined` (plus the calibration set's first 1,000) |
| 4 | `scripts/reproduce/04_align_reader.sh [500]` | Reader alignment on the calibration set (or its first 500 screenshots) and the acceptance test on the 1,007 screenshots of the unused calibration episodes | the calibration set's |
| 5 | `scripts/reproduce/05_evaluate.sh ADAPTER READER NAME` | One policy on the confirmation set (29 templates x seeds 3000-3011) with a reader, or in text | evaluation only |
| 6 | `scripts/reproduce/06_reader_analyses.sh` | Recurrence of misreads and new-episode predictability, CPU only | 0 |
| 7 | `$SIM_PY -m evaluation.bootstrap SPEC.json` | Paired bootstrap; see `configs/stats_main.example.json` | 0 |

The main comparison is `clean` (both seeds) with the aligned reader against `direct` (both seeds) with the original
reader. In the paper: 80.9% against 55.6% (+25.3 points, 95% interval [13.4, 38.1]); the aligned reader's element
error rate on the acceptance screenshots was 1.56%, against 25.62% for the original reader; alignment training took
93 minutes on one A6000. Model serving with batched requests is not bit-for-bit deterministic, so a rerun gives close
but not identical numbers. The bootstrap draws all comparisons of a spec from one random stream over its seed groups,
so its intervals also move slightly with the set of groups a spec lists.

Every reader call is recorded with the purpose it was spent on (`--purpose`: calibration, train, eval; re-reads are
booked as audit), so the screenshot budget of each method can be checked from the run directories.

## The aligned reader

The study's aligned reader (the LoRA adapter of step 4, trained on the 2,004 calibration screenshots) is released on
the Hugging Face Hub under CC BY-NC 4.0: https://huggingface.co/xuzhl/practice-in-text-aligned-reader. To use it
without training, merge it into the base reader and serve the merged directory as the reader:

```bash
$GPU_PY -m reader.merge_lora --base $MODELS_DIR/Qwen3-VL-4B-Instruct --adapter <downloaded adapter> --out $WORK_DIR/models/aligned_reader
```

## AndroidControl (optional)

`androidcontrol/` repeats the two reader-side measurements on AndroidControl, offline recordings of people using real
Android apps: whether misreads recur on the same elements, and how much alignment lowers the element error rate on
held-out screenshots. The text view comes from each step's accessibility tree, with every element checked against the
screenshot so that only visible content remains; unlike MobileGym's it has no colours and no icon names, and it leaves
out keyboards and web views. The reader, its prompt, the parser, the matching and the alignment recipe are MobileGym's.

The data is not included. `bash androidcontrol/download.sh` fetches the split files and shards 0-3 of the training data
(about 10 GB) from the public bucket into `$ANDROIDCONTROL_DIR/raw` and checks their md5; check the dataset's licence
first. Then run `scripts/reproduce/07_androidcontrol.sh` (sampling on the CPU, one GPU for reading and alignment). The
pixel checks need numpy and the font DejaVu Sans (`AC_FONT`, default `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`).

In the paper's supplement, misreads recurred 23.5 points more often than independent errors (the line was met), and
alignment lowered the element error rate from 36.68% to 27.45%, a relative drop of 25.2%, short of the 30% line.

## Run directories

`policy/episodes.py` and `reader/reread.py` write `$WORK_DIR/runs/<time>_<name>/`:

- `meta.json`: machine, process, code and MobileGym versions, all settings;
- `rows.jsonl`: one row per episode (`id` = template, `seed`, `sample`, `official_success` = judge accepted and the
  policy said it was done, `steps`, `stop`, ...);
- `steps/<template>_<seed>[_k<sample>].jsonl`: one record per step: `v0` (the true text view of the page),
  `reader_raw` (what the reader wrote, real pipeline only), `screen` (what the policy saw), `policy_user`,
  `policy_raw`, `action`, ...;
- `shots/`: the screenshots (PNG with `--png`, else JPEG);
- `vision_ledger.jsonl`: one line per reader call.

The analysis tools (`reader.error_rate`, `reader.recurrence`, `reader.align_data`, `practice.noise`,
`evaluation.bootstrap`) read these directories and need no GPU.

## Licence

The code in this repository is released under the Apache License 2.0 (`LICENSE`). Third-party software, models and
data used by it are listed in `THIRD_PARTY_NOTICES.md` with their own licences; none of them is redistributed here.
MobileGym's data, and anything generated from it, stays under CC BY-NC 4.0.

## Citation

```bibtex
@article{xu2026practice,
  title   = {Practice in Text, Act in Pixels: Correcting Where Perception Fails},
  author  = {Xu, Zhongling and Wang, Hairong},
  journal = {arXiv preprint arXiv:2611.22722},
  year    = {2026}
}
```
