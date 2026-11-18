<h1 align="center">Practice in Text, Act in Pixels: Correcting Where Perception Fails</h1>

<p align="center">
<a href="https://arxiv.org/abs/2611.22722"><img src="https://img.shields.io/badge/arXiv-2611.22722-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv paper"></a>
<a href="https://huggingface.co/xuzhl/practice-in-text-aligned-reader"><img src="https://img.shields.io/badge/Hugging%20Face-Aligned%20reader-ffd21e?logo=huggingface" alt="Aligned reader on Hugging Face"></a>
</p>

A GUI agent can practise cheaply on a text version of an app, then act on real screenshots through a *screen reader*:
a vision-language model that turns each screenshot into the same kind of text. The reader misreads elements and lists
them in its own order, so success drops sharply. *Reader alignment* fine-tunes the reader on about 2,000 screenshots
so that it writes exactly the text the agent practised on. The agent itself is never trained on screenshots.

On MobileGym, with the same 2,000 screenshots, an agent trained only in text and acting through the aligned reader
completes **80.9%** of 348 new task instances; an agent trained directly on screenshots completes **55.6%**.

## Requirements

- Linux with one NVIDIA GPU of 48 GB (we used an RTX A6000) and a driver for CUDA 12.8
- conda and git; about 40 GB of disk for models and environments, plus room for runs

## Setup

**1. Code and paths.** Clone the repository, copy the path template, edit its paths (MobileGym, work directory,
models, the two environments of step 2) and load it in every new shell:

```bash
git clone https://github.com/Carlxuzhl/practice-in-text.git && cd practice-in-text
cp configs/paths.env.example configs/paths.env
source configs/paths.env
```

**2. Two environments**, one for the simulator and the analyses, one for the GPU work, at the paths in the template:

```bash
conda create -y -p ~/envs/ptap-sim -c conda-forge python=3.11 'nodejs>=22,<23' pip
~/envs/ptap-sim/bin/pip install -r requirements/sim.txt
conda create -y -p ~/envs/ptap-gpu -c conda-forge python=3.11 pip
~/envs/ptap-gpu/bin/pip install -r requirements/gpu.txt
```

Keep Node.js out of the GPU environment: otherwise vLLM can fail on older Linux (e.g. Ubuntu 20.04) with
`CXXABI_1.3.15 not found`.

**3. MobileGym** (not included). The setup script clones [MobileGym](https://github.com/Purewhiter/mobilegym) at the
commit we used, builds the simulator and installs a headless Chromium. Keep the simulator served while you run
episodes:

```bash
bash scripts/setup_sim.sh
bash scripts/serve_sim.sh
```

Do not add MobileGym's optional dataset folder (`mobilegym-data`); we ran without it. MobileGym's code is Apache-2.0;
its data, and anything generated from it, is CC BY-NC 4.0 (non-commercial use).

**4. Models** (not included): Qwen3-8B (the agent) and Qwen3-VL-4B-Instruct (the reader), at the revisions pinned in
`configs/models.json`:

```bash
$GPU_PY scripts/download_models.py policy reader
```

The other two readers in that file (`reader_8b`, `reader_other_family`) are needed only for the comparisons with other
readers.

**5. Check the setup:**

```bash
$SIM_PY -m pytest -q tests      # CPU tests
bash scripts/smoke.sh           # one run through everything on the CPU, with stand-in models
bash scripts/smoke.sh --gpu 0   # the same with the real models on GPU 0 (about 15 minutes)
```

## Reproduce the study

Run from the repository root with `configs/paths.env` loaded, the simulator served (in another shell) and `GPU` set to
the card to use (`export GPU=0`). Each step prints the run directories it writes under `$WORK_DIR/runs/`.

```bash
bash scripts/reproduce/01_start_policy.sh       # scripted demonstrations -> the start agent
bash scripts/reproduce/02_calibration_set.sh    # 2,000 screenshots, each with its true text view
bash scripts/reproduce/03_practice.sh clean 0   # practice for one group and training seed
bash scripts/reproduce/04_align_reader.sh       # align the reader; test it on 1,007 held-out screenshots
bash scripts/reproduce/05_evaluate.sh $WORK_DIR/adapters/clean_s0_r2 $WORK_DIR/models/aligned_reader clean_s0_aligned
bash scripts/reproduce/06_reader_analyses.sh    # how often misreads recur (CPU only)
```

- **Step 3** trains one group with one training seed (0 or 1), in two practice rounds. Groups: `clean` (text),
  `generic` (text with random noise), `rate` (text with noise matched to the reader), `direct` (screenshots: round 1
  on the calibration set's first 1,000, round 2 on 1,000 new reads, so 2,000 in all) and `combined` (text with noise
  fitted on those first 1,000, then a round as in `direct`).
- **Step 4** aligns the reader on the whole calibration set and writes it to `$WORK_DIR/models/aligned_reader`;
  `04_align_reader.sh 500` uses only its first 500 screenshots.
- **Step 5** evaluates one agent with one reader on the 348 evaluation episodes (29 tasks, 12 new instances each). As
  the reader, give `$MODELS_DIR/Qwen3-VL-4B-Instruct` for the original one, or `text` to evaluate in text only.
- **Main comparison:** `clean` with the aligned reader against `direct` with the original reader, both with training
  seeds 0 and 1. Compare them with the paired bootstrap over tasks: `$SIM_PY -m evaluation.bootstrap SPEC.json`
  (example: `configs/stats_main.example.json`).

**Expected results** (from the paper): 80.9% against 55.6% (+25.3 points, 95% interval [13.4, 38.1]). On the held-out
screenshots the reader's element error rate falls from 25.62% to 1.56% after alignment. Alignment training takes
about 1.5 hours on one A6000. Model serving is not bit-for-bit deterministic, so a rerun gives close but not identical
numbers.

## Use the released aligned reader

The study's own aligned reader (the adapter of step 4; CC BY-NC 4.0) is on Hugging Face. Download it, merge it
into the reader, and pass the merged folder as the reader in step 5:

```bash
$GPU_PY scripts/download_models.py aligned_reader
$GPU_PY -m reader.merge_lora --base $MODELS_DIR/Qwen3-VL-4B-Instruct \
  --adapter $MODELS_DIR/practice-in-text-aligned-reader --out $WORK_DIR/models/aligned_reader
```

The reader must be called as in this code (prompt, greedy decoding, half-size screenshots; `configs/reader.json`).

## AndroidControl (optional)

The two reader measurements on recordings of real Android apps: whether misreads recur on the same elements, and how
much alignment lowers the element error rate on held-out screenshots. The text view comes from each screen's
accessibility tree, keeping only what the screenshot shows. The data is not included; the download script fetches
about 10 GB from the public bucket and checks it (see the dataset's licence first).

```bash
bash androidcontrol/download.sh
bash scripts/reproduce/07_androidcontrol.sh
```

Expected (paper's supplement): misreads recur 23.5 points more often than independent errors would; alignment lowers
the element error rate from 36.68% to 27.45%, a 25.2% relative drop, short of the 30% set in advance. The pixel checks
need the DejaVu Sans font (Debian/Ubuntu package `fonts-dejavu-core`; elsewhere set `AC_FONT` to the font file).

## What is in the repository

| Folder | Contents |
|---|---|
| `textview/` | Turns a MobileGym screen into the text view |
| `reader/` | Calls the reader, parses its output, measures its errors, aligns it |
| `practice/` | Text practice with added noise |
| `policy/` | Runs episodes; scripted demonstrations; trains the agent |
| `evaluation/` | Paired bootstrap |
| `androidcontrol/` | The AndroidControl experiment (no data) |
| `configs/` | Models, settings, tasks and seeds, path template |
| `scripts/` | Setup, model servers, the reproduction steps, smoke test |
| `tests/` | CPU tests |

Every run directory holds `rows.jsonl` (one line per episode; `official_success` is the result), `steps/` (per step:
the true text view `v0`, the reader's output `reader_raw`, the agent's input and action), `shots/` (screenshots) and
`vision_ledger.jsonl` (every reader call, with what it was spent on).

Not included: model weights, MobileGym and data generated from it, AndroidControl's data, the scripts for the paper's
tables and figures, and a few secondary analyses (the element-replay simulator and the gap decomposition, the look-up
reader, the same-label adaptation baseline).

## License

The code is under the Apache License 2.0 (`LICENSE`). Third-party software, models and data keep their own licences
(`THIRD_PARTY_NOTICES.md`); none of them is redistributed here.

## Citation

```bibtex
@article{xu2026practice,
  title   = {Practice in Text, Act in Pixels: Correcting Where Perception Fails},
  author  = {Xu, Zhongling and Wang, Hairong},
  journal = {arXiv preprint arXiv:2611.22722},
  year    = {2026}
}
```
