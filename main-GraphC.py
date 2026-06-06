# =========================================================
# Graph C reproduction - improved version
# מתאים לשחזור Figure C מהמאמר:
# Fisher overlap לפי עומק שכבה
# low = שתי פרמוטציות שונות של ריבוע 8x8 במרכז
# high = שתי פרמוטציות שונות של ריבוע 26x26 במרכז
# =========================================================

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import torchvision
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# לפי המאמר Figure C:
# 6 שכבות חבויות, רוחב 100, 100 epochs, ללא dropout וללא early stopping
# =========================================================
SEEDS = [0, 1, 2, 3, 4]

BATCH_SIZE = 256
FISHER_BATCH_SIZE = 1       # חשוב: Fisher יותר מדויק כשמחשבים פר דוגמה
WIDTH = 100
NUM_HIDDEN_LAYERS = 6

EPOCHS = 100
LR = 0.001
MOMENTUM = 0.9

VAL_FRACTION = 0.1

# כדי לא להפוך את Fisher לאיטי מדי.
# אפשר להעלות ל-10000 אם יש זמן.
FISHER_SUBSET_SIZE = 20000

NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# SEED
# =========================================================
def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA
# =========================================================
def load_base_mnist():
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    x_train = train_ds.data.float().div(255.0).unsqueeze(1)
    x_test = test_ds.data.float().div(255.0).unsqueeze(1)

    # נרמול סטנדרטי של MNIST
    x_train = (x_train - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081

    y_train = train_ds.targets.long()
    y_test = test_ds.targets.long()

    return x_train, y_train, x_test, y_test


BASE_X_TRAIN, BASE_Y_TRAIN, BASE_X_TEST, BASE_Y_TEST = load_base_mnist()


# =========================================================
# PERMUTATIONS
# =========================================================
def make_square_indices(square_size):
    """
    מחזיר את האינדקסים של ריבוע במרכז תמונת 28x28.
    למשל:
    square_size=8  -> ריבוע קטן במרכז
    square_size=26 -> כמעט כל התמונה
    """
    start = (28 - square_size) // 2
    end = start + square_size

    idxs = []
    for r in range(start, end):
        for c in range(start, end):
            idxs.append(r * 28 + c)

    return torch.tensor(idxs, dtype=torch.long)


def make_partial_square_permutation(square_size, seed):
    """
    יוצר פרמוטציה שבה רק הריבוע המרכזי עובר ערבוב.
    שאר הפיקסלים נשארים במקום.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.arange(784)
    square_idxs = make_square_indices(square_size)

    shuffled_square = square_idxs[torch.randperm(len(square_idxs), generator=g)]
    perm[square_idxs] = shuffled_square

    return perm


def apply_permutation(x, perm):
    """
    מפעיל את הפרמוטציה על התמונות.
    """
    flat = x.view(x.size(0), -1)
    flat = flat[:, perm]
    return flat.view(-1, 1, 28, 28)


def build_loaders_for_perm(perm, seed):
    """
    בונה loaders עבור משימה אחת:
    train / validation / test / fisher
    """
    x_train = apply_permutation(BASE_X_TRAIN, perm)
    x_test = apply_permutation(BASE_X_TEST, perm)

    full_train_ds = TensorDataset(x_train, BASE_Y_TRAIN)
    test_ds = TensorDataset(x_test, BASE_Y_TEST)

    val_size = int(len(full_train_ds) * VAL_FRACTION)
    train_size = len(full_train_ds) - val_size

    split_gen = torch.Generator()
    split_gen.manual_seed(seed + 999)

    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, val_size],
        generator=split_gen
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Fisher מחושב על subset כדי לחסוך זמן
    fisher_size = min(FISHER_SUBSET_SIZE, len(train_ds))
    fisher_subset = Subset(train_ds, list(range(fisher_size)))

    # batch_size=1 כדי לקבל קירוב טוב יותר ל-average per-sample squared gradient
    fisher_loader = DataLoader(
        fisher_subset,
        batch_size=FISHER_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader, fisher_loader


# =========================================================
# MODEL
# =========================================================
class SixLayerMLP(nn.Module):
    """
    רשת fully-connected עם 6 שכבות חבויות.
    זה מתאים לטבלת ההיפרפרמטרים של Figure C במאמר.
    """
    def __init__(self, width=WIDTH, num_hidden=NUM_HIDDEN_LAYERS):
        super().__init__()

        self.flatten = nn.Flatten()
        self.hidden_layers = nn.ModuleList()

        in_dim = 784
        for _ in range(num_hidden):
            self.hidden_layers.append(nn.Linear(in_dim, width))
            in_dim = width

        self.output = nn.Linear(width, 10)

    def forward(self, x):
        x = self.flatten(x)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return self.output(x)


# =========================================================
# TRAIN / EVAL
# =========================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_model(model, train_loader, val_loader, label):
    """
    אימון רגיל ל-100 epochs.
    אין early stopping, אבל כן נשמור את המודל עם ה-validation accuracy הכי טוב,
    כדי להימנע ממודל סופי גרוע במקרה של רעש באימון.
    """
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[{label}] Epoch {epoch+1}/{EPOCHS} "
            f"- loss={running_loss / len(train_loader):.4f} "
            f"- val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    return model


# =========================================================
# FISHER
# =========================================================
def compute_diag_fisher(model, loader):
    """
    מחשב diagonal Fisher:
    F_i ≈ average over samples of (dL/dtheta_i)^2

    חשוב:
    כאן loader עם batch_size=1, כדי שלא נרבע גרדיאנט ממוצע של batch שלם.
    """
    fisher = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    model.eval()
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

        sample_count += 1

    for n in fisher:
        fisher[n] /= max(sample_count, 1)

    return fisher


def fisher_hidden_layer_weight_vectors(fisher_dict):
    """
    מחלץ רק את ה-weight של כל שכבה חבויה.
    לא משתמשים ב-bias כי הוא יכול לטשטש את ההבדל בין השכבות.
    """
    layer_vectors = []

    for i in range(NUM_HIDDEN_LAYERS):
        name = f"hidden_layers.{i}.weight"

        if name not in fisher_dict:
            raise RuntimeError(f"Missing Fisher values for {name}")

        layer_vectors.append(fisher_dict[name].reshape(-1))

    return layer_vectors


def fisher_overlap_diag(f1, f2, eps=1e-12):
    """
    חישוב overlap עבור Fisher דיאגונלי.

    במאמר מגדירים:
    overlap = 1 - d^2
    עבור מטריצות דיאגונליות, זה שקול לסכום:
    sum sqrt(f1_norm * f2_norm)

    כלומר זה כמו Bhattacharyya coefficient בין שני וקטורי Fisher מנורמלים.
    """
    f1 = f1.clamp_min(0)
    f2 = f2.clamp_min(0)

    s1 = f1.sum()
    s2 = f2.sum()

    if s1.item() < eps or s2.item() < eps:
        return 0.0

    f1n = f1 / s1
    f2n = f2 / s2

    overlap = torch.sqrt(f1n * f2n).sum()
    return float(overlap.item())


def compute_layerwise_overlap(fisher_a, fisher_b):
    """
    מחשב overlap לכל אחת מ-6 השכבות.
    """
    layers_a = fisher_hidden_layer_weight_vectors(fisher_a)
    layers_b = fisher_hidden_layer_weight_vectors(fisher_b)

    overlaps = []
    for fa, fb in zip(layers_a, layers_b):
        overlaps.append(fisher_overlap_diag(fa, fb))

    return overlaps


# =========================================================
# RUN ONE SEED
# =========================================================
def train_task_and_compute_fisher(perm, seed, init_state, label):
    """
    מאמן מודל על משימה אחת ומחזיר Fisher.
    כל המשימות מתחילות מאותו initialization כדי שההשוואה תהיה הוגנת.
    """
    train_loader, val_loader, test_loader, fisher_loader = build_loaders_for_perm(perm, seed)

    model = SixLayerMLP().to(device)
    model.load_state_dict(init_state)

    model = train_model(model, train_loader, val_loader, label=label)

    acc = evaluate(model, test_loader)
    fisher = compute_diag_fisher(model, fisher_loader)

    return acc, fisher


def run_one_seed(seed):
    print("\n" + "=" * 80)
    print(f"RUNNING SEED {seed}")
    print("=" * 80)

    set_all_seeds(seed)

    # שתי פרמוטציות שונות עבור low ושתי פרמוטציות שונות עבור high
    # זה יותר קרוב לרעיון של השוואה בין שתי משימות שונות באותה דרגת שינוי.
    low_perm_1 = make_partial_square_permutation(square_size=8, seed=seed + 100)
    low_perm_2 = make_partial_square_permutation(square_size=8, seed=seed + 101)

    high_perm_1 = make_partial_square_permutation(square_size=26, seed=seed + 200)
    high_perm_2 = make_partial_square_permutation(square_size=26, seed=seed + 201)

    # אותה אתחול לכל המודלים באותו seed
    set_all_seeds(seed)
    init_model = SixLayerMLP().to(device)
    init_state = copy.deepcopy(init_model.state_dict())

    # Low pair
    acc_low_1, fisher_low_1 = train_task_and_compute_fisher(
        low_perm_1,
        seed + 1,
        init_state,
        label=f"seed{seed}_low_8x8_A"
    )

    acc_low_2, fisher_low_2 = train_task_and_compute_fisher(
        low_perm_2,
        seed + 2,
        init_state,
        label=f"seed{seed}_low_8x8_B"
    )

    # High pair
    acc_high_1, fisher_high_1 = train_task_and_compute_fisher(
        high_perm_1,
        seed + 3,
        init_state,
        label=f"seed{seed}_high_26x26_A"
    )

    acc_high_2, fisher_high_2 = train_task_and_compute_fisher(
        high_perm_2,
        seed + 4,
        init_state,
        label=f"seed{seed}_high_26x26_B"
    )

    overlap_low = compute_layerwise_overlap(fisher_low_1, fisher_low_2)
    overlap_high = compute_layerwise_overlap(fisher_high_1, fisher_high_2)

    print("\nSeed result:")
    print(f"Low 8x8 A accuracy:   {acc_low_1:.4f}")
    print(f"Low 8x8 B accuracy:   {acc_low_2:.4f}")
    print(f"High 26x26 A acc:     {acc_high_1:.4f}")
    print(f"High 26x26 B acc:     {acc_high_2:.4f}")
    print(f"Low overlap:          {overlap_low}")
    print(f"High overlap:         {overlap_high}")

    return np.array(overlap_low), np.array(overlap_high)


# =========================================================
# PLOT
# =========================================================
def plot_graph_c(overlap_low_avg, overlap_high_avg):
    x = list(range(1, NUM_HIDDEN_LAYERS + 1))

    plt.figure(figsize=(6.0, 4.2))

    plt.plot(
        x,
        overlap_low_avg,
        color="gray",
        linestyle="--",
        linewidth=2.2,
        marker="o",
        label="low % permutation"
    )

    plt.plot(
        x,
        overlap_high_avg,
        color="black",
        linestyle="--",
        linewidth=2.2,
        marker="o",
        label="high % permutation"
    )

    plt.xlabel("Layer depth")
    plt.ylabel("Overlap in Fisher")
    plt.xticks(x)
    plt.ylim(0.0, 1.0)
    plt.xlim(1, NUM_HIDDEN_LAYERS)
    plt.legend(frameon=False)
    plt.tight_layout()

    filename = "figure_C_improved.png"
    plt.savefig(filename, dpi=300)
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    total_start = time.time()

    print("\n========== Running improved Figure C reproduction ==========")
    print(f"SEEDS={SEEDS}")
    print(f"WIDTH={WIDTH}")
    print(f"NUM_HIDDEN_LAYERS={NUM_HIDDEN_LAYERS}")
    print(f"EPOCHS={EPOCHS}")
    print(f"LR={LR}")
    print(f"FISHER_SUBSET_SIZE={FISHER_SUBSET_SIZE}")
    print(f"FISHER_BATCH_SIZE={FISHER_BATCH_SIZE}")

    low_runs = []
    high_runs = []

    for seed in SEEDS:
        start = time.time()

        low_overlap, high_overlap = run_one_seed(seed)

        low_runs.append(low_overlap)
        high_runs.append(high_overlap)

        print(f"\nSeed {seed} finished in {(time.time() - start) / 60:.2f} min")

    low_avg = np.mean(np.stack(low_runs), axis=0)
    high_avg = np.mean(np.stack(high_runs), axis=0)

    print("\nAveraged layer-wise Fisher overlap:")
    print(f"Low 8x8:    {low_avg.tolist()}")
    print(f"High 26x26: {high_avg.tolist()}")

    plot_graph_c(low_avg, high_avg)

    print(f"\nTotal runtime: {(time.time() - total_start) / 60:.2f} min")
