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
├── graph A.py              # Figure 1A / 2A — Online EWC vs SGD on 3 sequential tasks
├── graph_B_version1.py     # Figure 2B  — EWC vs SGD+Dropout over 10 sequential tasks
├── GraphC.py               # Figure 2C  — Layer-wise Fisher overlap analysis
  
├── figure_2A_online_perfect.png      # Output: Figure 2A (Online EWC variant)
├── figure_2A_perfect.png             # Output: Figure 2A (alternate run)
├── graphC.png                        # Output: Figure 2C
├── Graph_B.jpeg                      # Output: Figure 2B
└── AI_LOGS/                          # Full AI interaction transcripts
```

---

## Scripts Overview

### `graph A.py` — Figure 1A / 2A: SGD vs Online EWC (3 Tasks)

Reproduces the core result of the paper: a head-to-head comparison of a vanilla SGD baseline against an **Online EWC** model trained sequentially on 3 Permuted MNIST tasks (20 epochs each).

**Architecture:** 2-hidden-layer MLP (784 → 400 → 400 → 10), ReLU activations  
**Online EWC:** Fisher matrices from successive tasks are averaged (running mean) rather than stored separately, avoiding parameter bloat across many tasks  
**Key hyperparameters:** `lr=0.1`, `λ=15000`, gradient clipping at `max_norm=10`  
**Output:** `figure_2A_online_perfect.png` — per-task accuracy curves for all 3 tasks across 60 total epochs, with vertical dashed lines marking task boundaries

---

### `graph_B_version1.py` — Figure 2B: Average Accuracy over 10 Tasks

Scales the comparison to **10 sequential tasks**, tracking the *average* accuracy across all tasks seen so far. Three conditions are compared:

| Condition | Model | Details |
|-----------|-------|---------|
| Single-task reference | MLP (per task) | One fresh model per task; upper-bound baseline |
| SGD + Dropout | DropoutMLP | Input: 20% dropout; hidden: 50% dropout |
| EWC | MLP | Accumulated Fisher penalty across all tasks |

**Architecture:** 2-hidden-layer MLP with width **1500** (much wider than Figure 1A)  
**Key hyperparameters:** `lr=0.001` (EWC/single), `lr=0.003` (Dropout), `momentum=0.9`, `λ=12000`, 100 epochs per task  
**Output:** `figure_B_lambda_12000.png` — average fraction correct vs. number of tasks (tasks 2–10)

---

### `GraphC.py` — Figure 2C: Layer-wise Fisher Overlap

Investigates *why* EWC works by measuring how much two tasks share the same important weights at each layer depth. Uses a **6-hidden-layer MLP** (width 100) and computes the diagonal Fisher Information Matrix for two task pairs:

- **Low overlap:** Two tasks each permuting only a small **8×8 centre patch** of the image
- **High overlap:** Two tasks each permuting a large **26×26 centre patch** (almost the full image)

Fisher overlap between tasks is computed as the Bhattacharyya coefficient between normalised Fisher vectors at each layer. Results are averaged over **5 random seeds**.

**Architecture:** 6-hidden-layer MLP (784 → 100 × 6 → 10)  
**Key hyperparameters:** `lr=0.001`, `momentum=0.9`, 100 epochs, `fisher_batch_size=1` (per-sample gradients for accurate Fisher estimation)  
**Output:** `figure_C_improved.png` — Fisher overlap vs. layer depth for low vs. high permutation conditions

---

## Requirements

```
Python 3.x
torch
torchvision
matplotlib
numpy
```

Install with:

```bash
pip install torch torchvision matplotlib numpy
```

---

## Running the Scripts

Each script is independent and can be run separately.

```bash
# Figure 1A / 2A — fast, good starting point (~minutes on CPU)
python "graph A.py"

# Figure 2B — slow due to 10 tasks × 100 epochs × 3 conditions
python graph_B_version1.py

# Figure 2C — very slow due to 5 seeds × 4 models × 100 epochs + per-sample Fisher
python GraphC.py
```

> **GPU recommended** for `graph_B_version1.py` and `GraphC.py`. Both scripts auto-detect CUDA and will use it if available.

---

## Outputs

| File | Script | Description |
|------|--------|-------------|
| `figure_2A_online_perfect.png` | `graph A.py` | Per-task accuracy curves, SGD vs Online EWC, 3 tasks |
| `figure_B_lambda_12000.png` | `graph_B_version1.py` | Avg accuracy vs tasks, EWC vs SGD+Dropout vs single-task |
| `figure_C_improved.png` | `GraphC.py` | Layer-wise Fisher overlap, low vs high permutation |

---

## Key Design Decisions

**Online EWC (graph A.py):** Rather than accumulating a separate Fisher matrix per task, this implementation maintains a single running-average Fisher. This avoids memory growth with more tasks while preserving knowledge across the task sequence.

**Fisher estimation (GraphC.py):** Uses `batch_size=1` (per-sample gradients) rather than batch-averaged gradients. This produces a more accurate estimate of the true diagonal Fisher, since squaring batch-averaged gradients underestimates parameter importance.

**Gradient clipping:** All three scripts apply `clip_grad_norm_(max_norm=10.0)` to prevent the gradient explosions that arise from the large EWC penalty terms during sequential training.

---

## AI Collaboration Log

Per the project requirements, the traditional written report is replaced by full documentation of the AI-assisted development process.

- **Log location:** `AI_LOGS/` directory in this repository
- **How AI was used:**
  - Analysed the mathematical and practical differences between Split MNIST and Permuted MNIST
  - Debugged gradient explosion issues that arose during sequential task training
  - Tuned hyperparameters (EWC lambda, learning rate) across all three scripts
  - Designed the Online EWC running-average Fisher update strategy
  - Formatted Matplotlib output to match academic publication standards

---
---

# פרויקט שחזור: התגברות על שכחה קטסטרופלית בעזרת EWC

> שחזור מלא של **Figures 1A, 2A ו-2C** מהמאמר של DeepMind:
> *"Overcoming catastrophic forgetting in neural networks"* (Kirkpatrick et al., 2017).

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
├── graph A.py              # Figure 1A / 2A — Online EWC מול SGD על 3 משימות עוקבות
├── graph_B_version1.py     # Figure 2B  — EWC מול SGD+Dropout על 10 משימות עוקבות
├── GraphC.py               # Figure 2C  — ניתוח חפיפת Fisher לפי עומק שכבה
├── figure_1A_publication_ready.png   # פלט: Figure 1A
├── figure_2A_online_perfect.png      # פלט: Figure 2A (Online EWC)
├── figure_2A_perfect.png             # פלט: Figure 2A (ריצה חלופית)
├── graphC.png                        # פלט: Figure 2C
├── Graph_B.jpeg                      # פלט: Figure 2B
└── AI_LOGS/                          # תיעוד מלא של שיחות עם AI
```

---

## תיאור הסקריפטים

### `graph A.py` — Figure 1A / 2A: SGD מול Online EWC (3 משימות)

משחזר את התוצאה המרכזית של המאמר: השוואה ישירה בין SGD רגיל לבין **Online EWC** על 3 משימות Permuted MNIST עוקבות (20 epochs לכל משימה).

**ארכיטקטורה:** MLP עם 2 שכבות חבויות (784 → 400 → 400 → 10), פעולת הפעלה ReLU  
**Online EWC:** מטריצות הפישר ממשימות עוקבות ממוצעות (ממוצע רץ) במקום לשמור אותן בנפרד — חוסך זיכרון לאורך משימות רבות  
**היפרפרמטרים מרכזיים:** `lr=0.1`, `λ=15000`, חיתוך גרדיאנטים ב-`max_norm=10`  
**פלט:** `figure_2A_online_perfect.png` — עקומות דיוק לכל משימה לאורך 60 epochs כולל, עם קווים מקווקווים המסמנים מעברים בין משימות

---

### `graph_B_version1.py` — Figure 2B: דיוק ממוצע על פני 10 משימות

מרחיב את ההשוואה ל-**10 משימות עוקבות**, ועוקב אחרי הדיוק הממוצע על כל המשימות שנלמדו עד כה. שלושה תנאים מושווים:

| תנאי | מודל | פרטים |
|------|------|--------|
| Single-task reference | MLP (נפרד לכל משימה) | גבול עליון — מודל רענן לכל משימה |
| SGD + Dropout | DropoutMLP | Dropout: 20% בכניסה, 50% בשכבות חבויות |
| EWC | MLP | פנלטי פישר מצטבר על כל המשימות |

**ארכיטקטורה:** MLP עם רוחב **1500** (רחב בהרבה מ-Figure 1A)  
**היפרפרמטרים מרכזיים:** `lr=0.001` (EWC/single), `lr=0.003` (Dropout), `momentum=0.9`, `λ=12000`, 100 epochs לכל משימה  
**פלט:** `figure_B_lambda_12000.png` — שבר נכון ממוצע כפונקציה של מספר המשימות (משימות 2–10)

---

### `GraphC.py` — Figure 2C: חפיפת Fisher לפי עומק שכבה

חוקר *למה* EWC עובד על ידי מדידת כמה שתי משימות חולקות אותן משקולות חשובות בכל עומק שכבה. משתמש ב-**MLP עם 6 שכבות חבויות** (רוחב 100) ומחשב את מטריצת Fisher האלכסונית עבור שני זוגות משימות:

- **חפיפה נמוכה:** שתי משימות שמערבבות רק **ריבוע 8×8 במרכז** התמונה
- **חפיפה גבוהה:** שתי משימות שמערבבות **ריבוע 26×26 במרכז** (כמעט כל התמונה)

חפיפת Fisher מחושבת כמקדם Bhattacharyya בין וקטורי Fisher מנורמלים בכל שכבה. התוצאות מממוצעות על **5 seeds אקראיים**.

**ארכיטקטורה:** MLP עם 6 שכבות חבויות (784 → 100×6 → 10)  
**היפרפרמטרים מרכזיים:** `lr=0.001`, `momentum=0.9`, 100 epochs, `fisher_batch_size=1` (גרדיאנטים פר-דוגמה לאמידת Fisher מדויקת)  
**פלט:** `figure_C_improved.png` — חפיפת Fisher כפונקציה של עומק השכבה, עבור תנאי חפיפה נמוכה וגבוהה

---

## דרישות מערכת

```
Python 3.x
torch
torchvision
matplotlib
numpy
```

התקנה:

```bash
pip install torch torchvision matplotlib numpy
```

---

## הוראות הרצה

כל סקריפט עצמאי וניתן להרצה בנפרד:

```bash
# Figure 1A / 2A — מהיר, נקודת התחלה טובה (~דקות על CPU)
python "graph A.py"

# Figure 2B — איטי (10 משימות × 100 epochs × 3 תנאים)
python graph_B_version1.py

# Figure 2C — איטי מאוד (5 seeds × 4 מודלים × 100 epochs + Fisher פר-דוגמה)
python GraphC.py
```

> **GPU מומלץ** עבור `graph_B_version1.py` ו-`GraphC.py`. שני הסקריפטים מזהים CUDA אוטומטית ומשתמשים בו אם זמין.

---

## פלטי התוכנית

| קובץ | סקריפט | תיאור |
|------|--------|-------|
| `figure_2A_online_perfect.png` | `graph A.py` | עקומות דיוק לכל משימה, SGD מול Online EWC |
| `figure_B_lambda_12000.png` | `graph_B_version1.py` | דיוק ממוצע כפונקציה של מספר משימות |
| `figure_C_improved.png` | `GraphC.py` | חפיפת Fisher לפי עומק שכבה |

---

## החלטות עיצוב מרכזיות

**Online EWC:** במקום לצבור מטריצת Fisher נפרדת לכל משימה, המימוש שומר ממוצע רץ אחד משותף. זה מונע גידול בצריכת הזיכרון עם מספר המשימות תוך שמירה על הידע הצבור.

**אמידת Fisher (GraphC.py):** משתמש ב-`batch_size=1` (גרדיאנטים פר-דוגמה) ולא בגרדיאנטים ממוצעים של batch. זה מייצר אמידה מדויקת יותר של Fisher האלכסוני האמיתי, כיוון שריבוע גרדיאנט ממוצע מעריך בחסר את חשיבות הפרמטרים.

**חיתוך גרדיאנטים:** כל שלושת הסקריפטים מפעילים `clip_grad_norm_(max_norm=10.0)` כדי למנוע התפוצצות גרדיאנטים הנגרמת מאיברי הענישה הגדולים של EWC במהלך אימון רציף.

---

## תיעוד שילוב בינה מלאכותית

בהתאם לדרישות הפרויקט, דוח הסיכום המסורתי הוחלף בתיעוד מלא של תהליך העבודה מול כלי AI.

- **מיקום התיעוד:** תיקיית `AI_LOGS/` בפרויקט זה
- **כיצד AI שימש בפרויקט:**
  - ניתוח מעמיק של ההבדלים בין Split MNIST ל-Permuted MNIST
  - פתרון בעיות התפוצצות גרדיאנטים במהלך אימון רציף
  - כיוון היפרפרמטרים (lambda של EWC, קצב למידה) בשלושת הסקריפטים
  - עיצוב אסטרטגיית ממוצע הפישר הרץ ב-Online EWC
  - עיצוב גרפים ב-Matplotlib ברמה אקדמית המתאימה לפרסום
