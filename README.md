# Overcoming Catastrophic Forgetting — EWC Reproduction Project

> A practical reproduction of the main qualitative results from Figures A, B, and C of Kirkpatrick et al. (2017), *"Overcoming catastrophic forgetting in neural networks"*.

---

## What This Project Demonstrates

This project demonstrates two core ideas from continual learning:

1. **Catastrophic Forgetting** — a neural network trained sequentially on new tasks can lose accuracy on previously learned tasks.
2. **Elastic Weight Consolidation (EWC)** — a regularisation method that reduces forgetting by penalising changes to parameters that were important for previous tasks, estimated using the Fisher Information Matrix.

The project reproduces three figure-style experiments using Permuted MNIST and PyTorch.

---

## Project Structure

```text
.
├── main-graph A.py             # Figure 2A — SGD vs Online EWC on 3 sequential tasks
├── main-graph_B.py             # Figure 2B — EWC vs SGD+Dropout over 10 sequential tasks
├── main-GraphC.py              # Figure 2C — layer-wise Fisher overlap analysis
│
├── requirements.txt            # Python dependencies
├── RESULTS.md                  # Summary of outputs, runtimes, seeds, and notes
│
├── figure_2A_online_perfect.png      # Canonical output for Figure 2A
├── figure_2A_perfect.png             # Alternate Figure 2A run
├── Graph_B.jpeg                      # Earlier generated preview for Figure 2B
└── graphC.png                        # Earlier generated preview for Figure 2C
```

Re-running Figure B and Figure C generates the canonical output files:

```text
figure_B_lambda_12000.png
figure_C_improved.png
```

---

## Requirements

Use Python 3.10 or newer if possible.

Install the required packages with:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` contains:

```text
torch
torchvision
matplotlib
numpy
```

The scripts use `torchvision.datasets.MNIST(..., download=True)`, so the first run requires an internet connection to download MNIST into the local `./data` folder.

Do not commit the `data/` folder to GitHub.

---

## Recommended Reproduction Order

Run the figures in this order:

1. **Figure 2A** — fastest and best first sanity check.
2. **Figure 2B** — slower; reproduces the 10-task average accuracy comparison.
3. **Figure 2C** — slowest; computes Fisher overlap across 5 seeds.

A GPU is strongly recommended for Figures 2B and 2C.

---

## Figure 2A — SGD vs Online EWC

### Purpose

This experiment compares a standard SGD baseline against an **Online EWC variant** across 3 sequential Permuted MNIST tasks.

The goal is to show that SGD forgets earlier tasks, while EWC preserves performance better after learning new tasks.

### Run command

```bash
python "main-graph A.py"
```

### Expected output file

```text
figure_2A_online_perfect.png
```

### Expected behaviour

The script should:

1. Download MNIST if needed.
2. Build 3 tasks:
   - Task 1: original MNIST.
   - Task 2: MNIST with one fixed pixel permutation.
   - Task 3: MNIST with another fixed pixel permutation.
3. Train a vanilla SGD baseline.
4. Train an Online EWC model.
5. Plot per-task accuracy curves across all 3 tasks.
6. Save the final figure as `figure_2A_online_perfect.png`.

### Important scientific note

Figure 2A uses an **Online EWC variant**, not the exact original per-task Fisher storage formulation.

Instead of storing one Fisher matrix per previous task, it stores one consolidated Fisher estimate updated as:

```text
Fisher_new = 0.5 * Fisher_old + 0.5 * Fisher_current_task
```

This keeps EWC memory cost constant with respect to the number of tasks.

Recommended presentation wording:

> Figure 2A uses an Online EWC variant that maintains a single consolidated Fisher estimate instead of one Fisher matrix per previous task, keeping memory cost constant as the number of tasks grows.

### Runtime estimate

```text
CPU: ~5–15 minutes
GPU: a few minutes
```

---

## Figure 2B — Average Accuracy over 10 Tasks

### Purpose

This experiment extends the comparison to 10 sequential Permuted MNIST tasks and plots average accuracy over all tasks seen so far.

It compares:

| Condition | Meaning |
|---|---|
| Single-task reference | A separate fresh model trained for each task; upper-bound baseline |
| SGD + Dropout | A sequential model using dropout regularisation |
| EWC | A sequential model using Fisher-based EWC regularisation |

### Run command

```bash
python main-graph_B.py
```

### Expected output file

```text
figure_B_lambda_12000.png
```

### Expected behaviour

The script should:

1. Download/load MNIST.
2. Generate 10 Permuted MNIST tasks.
3. Train a single-task reference model.
4. Train an SGD+Dropout sequential baseline.
5. Train an EWC sequential model.
6. Compute the average accuracy over tasks seen so far.
7. Save the final plot as `figure_B_lambda_12000.png`.

### Reproducibility note

Figure 2B currently uses one fixed seed:

```text
SEEDS = [0]
```

This is acceptable for a practical reproduction because the experiment is slow. For a stronger statistical result, run multiple seeds such as `[0, 1, 2]` or `[0, 1, 2, 3, 4]`.

### Runtime estimate

```text
CPU: ~2–4 hours
GPU: ~20–40 minutes
```

---

## Figure 2C — Layer-wise Fisher Overlap

### Purpose

This experiment investigates why EWC works by measuring how much two tasks share the same important parameters across different network layers.

It compares two task-pair conditions:

| Condition | Meaning |
|---|---|
| Low overlap | Two tasks permuting only a small 8×8 centre patch |
| High overlap | Two tasks permuting a large 26×26 centre patch |

The Fisher overlap is computed layer by layer using the Bhattacharyya coefficient between normalised Fisher vectors.

### Run command

```bash
python main-GraphC.py
```

### Expected output file

```text
figure_C_improved.png
```

### Expected behaviour

The script should:

1. Download/load MNIST.
2. Build low-overlap and high-overlap task pairs.
3. Train a 6-hidden-layer MLP from the same initialisation for each task pair.
4. Compute diagonal Fisher estimates using per-sample gradients.
5. Repeat the experiment over 5 seeds.
6. Average the overlap curves.
7. Save the final plot as `figure_C_improved.png`.

### Runtime estimate

```text
CPU: ~4–8 hours
GPU: ~1–2 hours
```

---

## Running on a Headless Server or GitHub Codespaces

If Matplotlib display windows cause issues, run with a non-interactive backend.

Linux/macOS:

```bash
MPLBACKEND=Agg python "main-graph A.py"
MPLBACKEND=Agg python main-graph_B.py
MPLBACKEND=Agg python main-GraphC.py
```

Windows PowerShell:

```powershell
$env:MPLBACKEND="Agg"; python "main-graph A.py"
$env:MPLBACKEND="Agg"; python main-graph_B.py
$env:MPLBACKEND="Agg"; python main-GraphC.py
```

---

## After Running the Experiments

After regenerating the figures, verify that the following files exist:

```text
figure_2A_online_perfect.png
figure_B_lambda_12000.png
figure_C_improved.png
```

Then commit the regenerated outputs:

```bash
git add figure_2A_online_perfect.png figure_B_lambda_12000.png figure_C_improved.png
git commit -m "Add regenerated experiment outputs"
git push
```

---

## Troubleshooting

### `ModuleNotFoundError`

Install dependencies again:

```bash
pip install -r requirements.txt
```

### MNIST download fails

Make sure the environment has internet access on the first run. After MNIST is downloaded once into `./data`, future runs can reuse it.

### Figure B or C takes too long

This is expected. Figure B trains multiple models across 10 tasks, and Figure C computes Fisher overlap across 5 seeds. Use a GPU if available.

### Output file is missing

Check the console for errors first. If the script completed but the expected file is missing, verify the exact `plt.savefig(...)` filename inside the relevant script.

---

## Summary of Outputs

| Figure | Script | Expected output | Notes |
|---|---|---|---|
| Figure 2A | `main-graph A.py` | `figure_2A_online_perfect.png` | Online EWC variant vs SGD |
| Figure 2B | `main-graph_B.py` | `figure_B_lambda_12000.png` | EWC vs SGD+Dropout vs single-task reference |
| Figure 2C | `main-GraphC.py` | `figure_C_improved.png` | Fisher overlap by layer depth |
