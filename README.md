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
├── run_all.py                  # Single entry point: python run_all.py --figure A/B/C/all
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

## Requirements

```
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

```
torch
torchvision
matplotlib
numpy
```

---

## Running the Scripts

Each script can be run via the unified entry point:

```bash
python run_all.py --figure A   # Figure 2A  (~5–15 min on CPU)
python run_all.py --figure B   # Figure 2B  (~2–4 h on CPU, ~20–40 min on GPU)
python run_all.py --figure C   # Figure 2C  (~4–8 h on CPU, ~1–2 h on GPU)
python run_all.py --figure all # All three figures in sequence
```

Or run scripts directly:

```bash
python "main-graph A.py"
python main-graph_B.py
python main-GraphC.py
```

> **GPU recommended** for `main-graph_B.py` and `main-GraphC.py`. Both scripts auto-detect CUDA and will use it if available.

---

## Outputs

| File | Script | Description |
| ---- | ------ | ----------- |
| `figure_2A_online_perfect.png` | `main-graph A.py` | Per-task accuracy curves, SGD vs Online EWC, 3 tasks |
| `figure_B_lambda_12000.png` | `main-graph_B.py` | Avg accuracy vs tasks, EWC vs SGD+Dropout vs single-task |
| `figure_C_improved.png` | `main-GraphC.py` | Layer-wise Fisher overlap, low vs high permutation |

> `Graph_B.jpeg` and `graphC.png` in the repository are outputs from earlier runs. Re-running the scripts produces the canonical filenames above.

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
├── run_all.py                  # נקודת הרצה אחת: python run_all.py --figure A/B/C/all
├── requirements.txt            # תלויות Python
├── RESULTS.md                  # סיכום פלטים, זמני ריצה והערות שחזוריות
├── figure_2A_online_perfect.png      # פלט: Figure 2A (Online EWC)
├── figure_2A_perfect.png             # פלט: Figure 2A (ריצה חלופית)
├── Graph_B.jpeg                      # פלט: Figure 2B (ריצה קודמת)
├── graphC.png                        # פלט: Figure 2C (ריצה קודמת)
└── ChatGPT_Prompts.png               # תיעוד שיחות עם AI (צילום מסך)
```

---

## תיאור הסקריפטים

### `main-graph A.py` — Figure A: SGD מול Online EWC (3 משימות)

משחזר את התוצאה המרכזית של המאמר: השוואה ישירה בין SGD רגיל לבין **Online EWC** על 3 משימות Permuted MNIST עוקבות (20 epochs לכל משימה).

**ארכיטקטורה:** MLP עם 2 שכבות חבויות (784 → 400 → 400 → 10), פעולת הפעלה ReLU  
**Online EWC:** מטריצות הפישר ממוצעות (50/50) בין משימות — זיכרון קבוע ללא קשר למספר המשימות  
**היפרפרמטרים מרכזיים:** `lr=0.1`, `λ=15000`, חיתוך גרדיאנטים ב-`max_norm=10`  
**פלט:** `figure_2A_online_perfect.png`

---

### `main-graph_B.py` — Figure 2B: דיוק ממוצע על פני 10 משימות

מרחיב את ההשוואה ל-**10 משימות עוקבות**, ועוקב אחרי הדיוק הממוצע על כל המשימות שנלמדו עד כה. שלושה תנאים מושווים:

| תנאי                  | מודל                 | פרטים                                   |
| --------------------- | -------------------- | --------------------------------------- |
| Single-task reference | MLP (נפרד לכל משימה) | גבול עליון — מודל רענן לכל משימה        |
| SGD + Dropout         | DropoutMLP           | Dropout: 20% בכניסה, 50% בשכבות חבויות  |
| EWC                   | MLP                  | פנלטי Fisher מצטבר (חיבורי) על כל המשימות |

**ארכיטקטורה:** MLP עם רוחב **1500**  
**מימוש EWC:** `ConsolidatedEWC` — Fisher **מצטבר** (חיבורי) בין משימות  
**היפרפרמטרים מרכזיים:** `lr=0.001` (EWC/single), `lr=0.003` (Dropout), `momentum=0.9`, `λ=12000`, 100 epochs לכל משימה  
**פלט:** `figure_B_lambda_12000.png`

---

### `main-GraphC.py` — Figure 2C: חפיפת Fisher לפי עומק שכבה

חוקר *למה* EWC עובד על ידי מדידת כמה שתי משימות חולקות אותן משקולות חשובות בכל עומק שכבה.

- **חפיפה נמוכה:** שתי משימות שמערבבות רק **ריבוע 8×8 במרכז** התמונה
- **חפיפה גבוהה:** שתי משימות שמערבבות **ריבוע 26×26 במרכז**

**ארכיטקטורה:** MLP עם 6 שכבות חבויות (784 → 100×6 → 10)  
**היפרפרמטרים מרכזיים:** `lr=0.001`, `momentum=0.9`, 100 epochs, `fisher_batch_size=1`, 5 seeds  
**פלט:** `figure_C_improved.png`

---

## דרישות מערכת

```
pip install -r requirements.txt
```

---

## הוראות הרצה

```bash
python run_all.py --figure A    # Figure 2A  (~5–15 דקות על CPU)
python run_all.py --figure B    # Figure 2B  (~2–4 שעות על CPU)
python run_all.py --figure C    # Figure 2C  (~4–8 שעות על CPU)
python run_all.py --figure all  # כל שלושת הגרפים ברצף
```

> **GPU מומלץ** עבור `main-graph_B.py` ו-`main-GraphC.py`. שני הסקריפטים מזהים CUDA אוטומטית.

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

**Consolidated EWC (`main-graph_B.py`):** Fisher **מצטבר** (חיבורי) בין משימות — כל task מוסיף את ה-Fisher שלו על גבי הסך הקיים, בהתאם לנוסחה המקורית במאמר.

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
