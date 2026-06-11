# Results Summary

Reproduction of Kirkpatrick et al. (2017) — *Overcoming catastrophic forgetting in neural networks*.

## Output Files

| Figure | Script | Output file | Runtime estimate | Seeds | Notes |
|--------|--------|-------------|-----------------|-------|-------|
| 2A | `main-graph A.py` | `figure_2A_online_perfect.png` | ~5–15 min (CPU) | fixed (`42`) | Online EWC variant: single running-average Fisher (50/50 blend per task boundary). Memory cost is constant w.r.t. number of tasks. Not identical to the original per-task Fisher storage, but produces the same qualitative result. |
| 2B | `main-graph_B.py` | `figure_B_lambda_12000.png` | ~2–4 h (CPU) / ~20–40 min (GPU) | `[0]` | ConsolidatedEWC: Fisher is *accumulated* (additive) across tasks. Single seed; add `SEEDS = [0, 1, 2]` for more reliable error bars. |
| 2C | `main-GraphC.py` | `figure_C_improved.png` | ~4–8 h (CPU) / ~1–2 h (GPU) | 5 random seeds | 6-layer MLP, per-sample Fisher (`batch_size=1`). Low-overlap vs high-overlap patch permutations. |

## Existing output files in repo

The repository already contains two output images from earlier runs:

| File in repo | Corresponds to |
|---|---|
| `Graph_B.jpeg` | Figure 2B (earlier run of `main-graph_B.py`) |
| `graphC.png` | Figure 2C (earlier run of `main-GraphC.py`) |

Running the scripts again will produce the canonical output filenames listed in the table above (`figure_B_lambda_12000.png` and `figure_C_improved.png`).

## EWC variant note (Figure A)

Figure A uses an **Online EWC variant**, not the exact formulation from the original paper.
The original paper stores one Fisher matrix per previous task.
This implementation stores a single consolidated estimate updated as:

```
fisher_new = 0.5 * fisher_old + 0.5 * fisher_current_task
```

This keeps memory usage **constant** regardless of the number of tasks, at the cost of down-weighting older tasks over time. The qualitative result — EWC preserving past performance while SGD forgets — is fully reproduced.

Recommended framing for presentations:

> *Figure A uses an Online EWC variant that maintains a single consolidated Fisher estimate instead of one matrix per previous task, keeping memory cost O(1) with respect to task count.*
