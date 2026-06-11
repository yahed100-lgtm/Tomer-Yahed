# Overcoming Catastrophic Forgetting — EWC Reproduction Project

> A full reproduction of **Figures A, B, and C** from Kirkpatrick et al. (2017),
> *"Overcoming catastrophic forgetting in neural networks"* (DeepMind).

---

## What This Project Does

This project demonstrates two things side by side:

1. **Catastrophic Forgetting** — how a standard neural network loses accuracy on a previously learned task the moment it starts learning a new one.
2. **Elastic Weight Consolidation (EWC)** — how a biologically-inspired regularisation algorithm prevents that forgetting by protecting the weights that matter most.

Three separate figures from the original paper are reproduced, each from its own script.

---

## Why We Chose This Paper

Most solutions to catastrophic forgetting are purely mathematical. What sets this paper apart is that it is rooted in a genuinely compelling biological idea: the authors studied how the human brain preserves important memories while still acquiring new ones, and translated that principle into a concrete algorithm.

The key intuition is a **spring analogy**: critical network parameters are given a restoring force that resists large changes during new-task training. Instead of letting new learning silently overwrite old knowledge, EWC identifies the weights that mattered for past tasks — using the Fisher Information Matrix — and penalises moving them too far.

The combination of **machine learning, mathematics, and biological inspiration** is what makes this paper stand out. To us, there is something especially compelling about learning from the most effective learning system in nature — the human brain — and formalising its principles as an algorithm.

---

## Project Structure

```
.
├── main-graph A.py             # Figure 2A — Online EWC vs SGD on 3 sequential tasks
├── main-graph_B.py             # Figure 2B — EWC vs SGD+Dropout over 10 sequential tasks
├── main-GraphC.py              # Figure 2C — Layer-wise Fisher overlap analysis
│
├── requirements.txt            # Python dependencies
├── RESULTS.md                  # Output summary, runtimes, and reproducibility notes
│
├── figure_2A_online_perfect.png      # Output: Figure 2A (Online EWC variant)
├── figure_2A_perfect.png             # Output: Figure 2A (alternate run)
├── Graph_B.jpeg                      # Output: Figure 2B (earlier run)
├── graphC.png                        # Output: Figure 2C (earlier run)
└── ChatGPT_Prompts.png               # AI interaction log (screenshot)
```

---

## Scripts Overview

### `main-graph A.py` — Figure A: SGD vs Online EWC (3 Tasks)

Reproduces the core result of the paper: a head-to-head comparison of a vanilla SGD baseline against an **Online EWC** model trained sequentially on 3 Permuted MNIST tasks (20 epochs each).

**Architecture:** 2-hidden-layer MLP (784 → 400 → 400 → 10), ReLU activations  
**Online EWC:** Fisher matrices from successive tasks are combined via a running average (50/50 blend per task boundary) rather than stored separately, keeping memory cost constant with respect to the number of tasks  
**Key hyperparameters:** `lr=0.1`, `λ=15000`, gradient clipping at `max_norm=10`  
**Output:** `figure_2A_online_perfect.png` — per-task accuracy curves for all 3 tasks across 60 total epochs, with vertical dashed lines marking task boundaries

---

### `main-graph_B.py` — Figure 2B: Average Accuracy over 10 Tasks

Scales the comparison to **10 sequential tasks**, tracking the *average* accuracy across all tasks seen so far. Three conditions are compared:

| Condition             | Model          | Details                                        |
| --------------------- | -------------- | ---------------------------------------------- |
| Single-task reference | MLP (per task) | One fresh model per task; upper-bound baseline |
| SGD + Dropout         | DropoutMLP     | Input: 20% dropout; hidden: 50% dropout        |
| EWC                   | MLP            | Accumulated Fisher penalty across all tasks    |

**Architecture:** 2-hidden-layer MLP with width **1500** (much wider than Figure 2A)  
**EWC implementation:** `ConsolidatedEWC` — Fisher matrices are *accumulated* (added together) across tasks, so earlier tasks maintain persistent influence throughout training  
**Key hyperparameters:** `lr=0.001` (EWC/single), `lr=0.003` (Dropout), `momentum=0.9`, `λ=12000`, 100 epochs per task  
**Output:** `figure_B_lambda_12000.png`

---

### `main-GraphC.py` — Figure 2C: Layer-wise Fisher Overlap

Investigates *why* EWC works by measuring how much two tasks share the same important weights at each layer depth. Uses a **6-hidden-layer MLP** (width 100) and computes the diagonal Fisher Information Matrix for two task pairs:

- **Low overlap:** Two tasks each permuting only a small **8×8 centre patch** of the image
- **High overlap:** Two tasks each permuting a large **26×26 centre patch** (almost the full image)

Fisher overlap between tasks is computed as the Bhattacharyya coefficient between normalised Fisher vectors at each layer. Results are averaged over **5 random seeds**.

**Architecture:** 6-hidden-layer MLP (784 → 100 × 6 → 10)  
**Key hyperparameters:** `lr=0.001`, `momentum=0.9`, 100 epochs, `fisher_batch_size=1` (per-sample gradients for accurate Fisher estimation)  
**Output:** `figure_C_improved.png`

---

## Installation

**Python 3.10 or higher is required.**

Install all dependencies with:

```bash
pip install -r requirements.txt
```

For GPU support (CUDA 11.8), replace the torch/torchvision lines with:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy>=1.24.0 matplotlib>=3.7.0
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for other CUDA versions.

---

## Running the Scripts

Each script is fully independent. Run them directly from the project root:

**Figure 2A** — fast, ~5–15 minutes on CPU:
```bash
python "main-graph A.py"
```
Produces: `figure_2A_online_perfect.png`

**Figure 2B** — slow, ~2–4 hours on CPU / ~20–40 minutes on GPU:
```bash
python main-graph_B.py
```
Produces: `figure_B_lambda_12000.png`

**Figure 2C** — very slow, ~4–8 hours on CPU / ~1–2 hours on GPU:
```bash
python main-GraphC.py
```
Produces: `figure_C_improved.png`

> Both `main-graph_B.py` and `main-GraphC.py` auto-detect CUDA and will use the GPU if available. MNIST data (~11 MB) is downloaded automatically on first run into a `./data/` folder.

---

## Outputs

| File | Script | Description |
| ---- | ------ | ----------- |
| `figure_2A_online_perfect.png` | `main-graph A.py` | Per-task accuracy curves, SGD vs Online EWC, 3 tasks |
| `figure_B_lambda_12000.png` | `main-graph_B.py` | Avg accuracy vs tasks, EWC vs SGD+Dropout vs single-task |
| `figure_C_improved.png` | `main-GraphC.py` | Layer-wise Fisher overlap, low vs high permutation |

> `Graph_B.jpeg` and `graphC.png` already in the repository are outputs from earlier runs. Re-running the scripts produces the canonical filenames listed above.

---

## Key Design Decisions

**Online EWC (`main-graph A.py`):** Rather than accumulating a separate Fisher matrix per task, this implementation maintains a single running-average Fisher (50/50 blend at each task boundary). This avoids memory growth with more tasks while preserving knowledge across the task sequence. This is an Online EWC variant — not identical to the original per-task formulation, but reproduces the same qualitative result.

**Consolidated EWC (`main-graph_B.py`):** Uses additive Fisher accumulation — each new task's Fisher is added on top of the existing total. This means earlier tasks exert a stronger cumulative restraint on parameters, matching the original paper's formulation more closely.

**Fisher estimation (`main-GraphC.py`):** Uses `batch_size=1` (per-sample gradients) rather than batch-averaged gradients. This produces a more accurate estimate of the true diagonal Fisher, since squaring batch-averaged gradients underestimates parameter importance.

**Gradient clipping:** All three scripts apply `clip_grad_norm_(max_norm=10.0)` to prevent the gradient explosions that arise from the large EWC penalty terms during sequential training.

---

## AI Collaboration Log

Per the project requirements, the traditional written report is replaced by full documentation of the AI-assisted development process.

- **Log location:** `ChatGPT_Prompts.png` in this repository
- **How AI was used:**
  * Tuned hyperparameters (EWC lambda, learning rate) across all three scripts
  * Designed the Online EWC running-average Fisher update strategy
  * Formatted Matplotlib output to match academic publication standards

---

# פרויקט שחזור: התגברות על שכחה קטסטרופלית בעזרת EWC

> שחזור מלא של **Figures A, B ו-C** מהמאמר של DeepMind: *"Overcoming catastrophic forgetting in neural networks"* (Kirkpatrick et al., 2017).

---

## מה הפרויקט מדגים

הפרויקט מציג שני דברים זה לצד זה:

1. **שכחה קטסטרופלית** — כיצד רשת נוירונים סטנדרטית מאבדת דיוק על משימה קודמת ברגע שהיא מתחילה ללמוד משימה חדשה.
2. **Elastic Weight Consolidation (EWC)** — כיצד אלגוריתם מבוסס השראה ביולוגית מונע את השכחה הזו על ידי הגנה על המשקולות החשובות ברשת.

שלושה גרפים שונים מהמאמר המקורי משוחזרים, כל אחד מקובץ סקריפט נפרד.

---

## למה בחרנו במאמר הזה

רוב הפתרונות לשכחה קטסטרופלית הם מתמטיים גרידא. מה שהופך את המאמר הזה לייחודי הוא שהוא נשען על רעיון ביולוגי עמוק ומרתק: הכותבים בחנו כיצד המוח האנושי מצליח לשמר זיכרונות חשובים לאורך זמן גם תוך כדי למידת מידע חדש, והפכו את העיקרון הזה לאלגוריתם חישובי.

האינטואיציה המרכזית היא **דימוי הקפיץ**: לפרמטרים קריטיים ברשת ניתן כוח מחזיר המונע מהם להשתנות יותר מדי בזמן למידת משימות חדשות. במקום לאפשר ללמידה חדשה לדרוס ידע קודם, EWC מזהה את המשקולות שהיו חשובות למשימות הקודמות — באמצעות מטריצת המידע של פישר — ומעניש סטייה גדולה מהן.

החיבור בין **למידת מכונה, מתמטיקה והשראה ביולוגית מהמוח האנושי** הוא מה שהופך את המאמר לייחודי בעינינו.

---

## מבנה הפרויקט

```
.
├── main-graph A.py             # Figure 2A — Online EWC מול SGD על 3 משימות עוקבות
├── main-graph_B.py             # Figure 2B — EWC מול SGD+Dropout על 10 משימות עוקבות
├── main-GraphC.py              # Figure 2C — ניתוח חפיפת Fisher לפי עומק שכבה
├── requirements.txt            # תלויות Python
├── RESULTS.md                  # סיכום פלטים, זמני ריצה והערות שחזוריות
├── figure_2A_online_perfect.png      # פלט: Figure 2A (Online EWC)
├── figure_2A_perfect.png             # פלט: Figure 2A (ריצה חלופית)
├── Graph_B.jpeg                      # פלט: Figure 2B (ריצה קודמת)
├── graphC.png                        # פלט: Figure 2C (ריצה קודמת)
└── ChatGPT_Prompts.png               # תיעוד שיחות עם AI (צילום מסך)
```

---

## התקנה

**נדרש Python 3.10 ומעלה.**

```bash
pip install -r requirements.txt
```

להתקנה עם GPU (CUDA 11.8):

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy>=1.24.0 matplotlib>=3.7.0
```

לגרסאות CUDA אחרות ראו [pytorch.org](https://pytorch.org/get-started/locally/).

---

## הוראות הרצה

כל סקריפט עצמאי לחלוטין. מריצים ישירות מתיקיית הפרויקט:

**Figure 2A** — מהיר, ~5–15 דקות על CPU:
```bash
python "main-graph A.py"
```
מייצר: `figure_2A_online_perfect.png`

**Figure 2B** — איטי, ~2–4 שעות על CPU / ~20–40 דקות על GPU:
```bash
python main-graph_B.py
```
מייצר: `figure_B_lambda_12000.png`

**Figure 2C** — איטי מאוד, ~4–8 שעות על CPU / ~1–2 שעות על GPU:
```bash
python main-GraphC.py
```
מייצר: `figure_C_improved.png`

> `main-graph_B.py` ו-`main-GraphC.py` מזהים CUDA אוטומטית ומשתמשים ב-GPU אם זמין. נתוני MNIST (~11 MB) מורדים אוטומטית בהרצה הראשונה לתיקייה `./data/`.

---

## פלטי התוכנית

| קובץ | סקריפט | תיאור |
| ---- | ------- | ----- |
| `figure_2A_online_perfect.png` | `main-graph A.py` | עקומות דיוק לכל משימה, SGD מול Online EWC |
| `figure_B_lambda_12000.png` | `main-graph_B.py` | דיוק ממוצע כפונקציה של מספר משימות |
| `figure_C_improved.png` | `main-GraphC.py` | חפיפת Fisher לפי עומק שכבה |

> `Graph_B.jpeg` ו-`graphC.png` בריפו הם פלטים מריצות קודמות. הרצה מחדש תייצר את שמות הקבצים הנכונים לפי הטבלה.

---

## החלטות עיצוב מרכזיות

**Online EWC (`main-graph A.py`):** ממוצע רץ (50/50) — זיכרון קבוע ללא קשר למספר המשימות. זוהי וריאנט של EWC, לא המימוש המדויק מהמאמר המקורי, אך מייצרת את אותה תוצאה איכותית.

**Consolidated EWC (`main-graph_B.py`):** Fisher מצטבר (חיבורי) בין משימות — כל task מוסיף את ה-Fisher שלו על גבי הסך הקיים, בהתאם לנוסחה המקורית במאמר.

**אמידת Fisher (`main-GraphC.py`):** `batch_size=1` — גרדיאנטים פר-דוגמה לאמידה מדויקת יותר של Fisher האלכסוני.

**חיתוך גרדיאנטים:** כל שלושת הסקריפטים מפעילים `clip_grad_norm_(max_norm=10.0)`.

---

## תיעוד שילוב בינה מלאכותית

בהתאם לדרישות הפרויקט, דוח הסיכום המסורתי הוחלף בתיעוד מלא של תהליך העבודה מול כלי AI.

- **מיקום התיעוד:** `ChatGPT_Prompts.png` בפרויקט זה
- **כיצד AI שימש בפרויקט:**
  * כיוון היפרפרמטרים (lambda של EWC, קצב למידה) בשלושת הסקריפטים
  * עיצוב אסטרטגיית ממוצע הפישר הרץ ב-Online EWC
  * עיצוב גרפים ב-Matplotlib ברמה אקדמית המתאימה לפרסום
