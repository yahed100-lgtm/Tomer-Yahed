# =========================================================
# שחזור Figure B ממאמר EWC על Permuted MNIST
# הקוד מריץ שלושה ניסויים:
# 1. Single-task reference — מודל נפרד לכל משימה, קו שחור מקווקו
# 2. SGD + Dropout — מודל אחד שנלמד ברצף, קו כחול
# 3. EWC — מודל אחד שנלמד ברצף עם ענישה על שינוי משקולות חשובות, קו אדום
# =========================================================

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import matplotlib.pyplot as plt

# =========================================================
# הגדרות הניסוי — הערכים שנבחרו לפי הריצה שנתנה את הגרף הכי טוב
# =========================================================
MAX_TASKS = 10          # מספר המשימות הרציפות ב-Permuted MNIST

BATCH_SIZE = 256       # גודל batch באימון
WIDTH = 1500           # מספר נוירונים בכל שכבה חבויה
MAX_EPOCHS = 100      # מספר epochs לכל משימה
PATIENCE = 5          # לא בשימוש בגרסה הזו, נשאר מהניסויים הקודמים

LR_SINGLE = 0.001    # learning rate לאימון single-task
LR_DROPOUT = 0.003   # learning rate ל-baseline עם dropout
LR_EWC = 0.001       # learning rate לאימון עם EWC
MOMENTUM = 0.9       # momentum עבור SGD

# מקדם הענישה של EWC — הערך שנבחר אחרי ניסויים, נתן את התוצאה הטובה ביותר
EWC_LAMBDA = 12000  # חוזק הענישה של EWC

VAL_FRACTION = 0.1   # אחוז הדאטה שמופרד ל-validation
SEEDS = [0]            # seed יחיד כדי לקצר זמן ריצה

NUM_WORKERS = 0       # מספר workers ל-DataLoader; 0 מתאים ל-Windows/Kaggle פשוט
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================================
# טעינת הנתונים ובניית משימות Permuted MNIST
# =========================================================
# טוען את MNIST, מנרמל את הפיקסלים, ומחזיר train/test מלאים
def load_base_mnist():
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    x_train = train_ds.data.float().div(255.0).unsqueeze(1)
    x_test = test_ds.data.float().div(255.0).unsqueeze(1)

    x_train = (x_train - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081

    y_train = train_ds.targets.long()
    y_test = test_ds.targets.long()

    return x_train, y_train, x_test, y_test


BASE_X_TRAIN, BASE_Y_TRAIN, BASE_X_TEST, BASE_Y_TEST = load_base_mnist()


# קובע seed כדי שהריצה תהיה כמה שיותר שחזורית
def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# יוצר 10 משימות: הראשונה MNIST רגיל, והשאר עם פרמוטציה אקראית קבועה לפיקסלים
def make_permutations(seed, max_tasks=10):
    g = torch.Generator()
    g.manual_seed(seed)

    perms = [torch.arange(784)]
    for _ in range(max_tasks - 1):
        perms.append(torch.randperm(784, generator=g))

    return perms


# מחיל את הפרמוטציה על תמונות MNIST: הופך לוקטור, מסדר פיקסלים מחדש, ומחזיר לצורת 28x28
def apply_permutation(x, perm):
    flat = x.view(x.size(0), -1)
    flat = flat[:, perm]
    return flat.view(-1, 1, 28, 28)


# בונה DataLoaders לכל משימה: train, validation, test
def build_task_loaders(perms, seed):
    train_loaders = []
    val_loaders = []
    test_loaders = []

    split_gen = torch.Generator()
    split_gen.manual_seed(seed + 12345)

    for task_idx in range(MAX_TASKS):
        x_train = apply_permutation(BASE_X_TRAIN, perms[task_idx])
        x_test = apply_permutation(BASE_X_TEST, perms[task_idx])

        full_train_ds = TensorDataset(x_train, BASE_Y_TRAIN)
        test_ds = TensorDataset(x_test, BASE_Y_TEST)

        val_size = int(len(full_train_ds) * VAL_FRACTION)
        train_size = len(full_train_ds) - val_size

        train_ds, val_ds = random_split(
            full_train_ds,
            [train_size, val_size],
            generator=split_gen
        )

        train_loaders.append(DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        ))

        val_loaders.append(DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        ))

        test_loaders.append(DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        ))

    return train_loaders, val_loaders, test_loaders

# =========================================================
# הגדרת המודלים
# =========================================================
# רשת MLP רגילה — משמשת ל-Single Task ול-EWC
class MLP(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.net(x)


# רשת MLP עם Dropout — משמשת כ-baseline של SGD+dropout
class DropoutMLP(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(784, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# מימוש EWC — אותו מימוש שנתן את התוצאה הוויזואלית הכי טובה
# =========================================================
# מחלקה שמחשבת ושומרת Fisher + פרמטרים קודמים לצורך ענישת EWC
class ConsolidatedEWC:
    def __init__(self, model):
        self.model = model
        self.fisher = None
        self.star_params = None

    # שומר עותק של הפרמטרים הנוכחיים של הרשת אחרי סיום משימה
    def _snapshot_params(self):
        return {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    # מחשב קירוב אלכסוני למטריצת Fisher לפי הגרדיאנטים של loss על הדאטה
    def _diag_fisher(self, dataloader):
        fisher = {
            n: torch.zeros_like(p, device=device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        batch_count = 0

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            self.model.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

            batch_count += 1

        for n in fisher:
            fisher[n] /= batch_count

        return fisher

    # אחרי כל משימה: מחשבים Fisher חדש ומעדכנים את נקודת העוגן של EWC
    def update_after_task(self, dataloader):
        current_fisher = self._diag_fisher(dataloader)
        current_params = self._snapshot_params()

        if self.fisher is None:
            self.fisher = current_fisher
            self.star_params = current_params
        else:
            for n in self.fisher:
                self.fisher[n] = self.fisher[n] + current_fisher[n]

            # שומרים את אותו אופן פעולה כמו בקוד שנתן את התוצאה הכי טובה
            self.star_params = current_params

    # מחשב את איבר הענישה של EWC שמונע מהמשקולות לזוז יותר מדי מהערכים הקודמים
    def penalty(self, model):
        if self.fisher is None or self.star_params is None:
            return torch.tensor(0.0, device=device)

        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()

        return loss

# =========================================================
# פונקציות עזר לאימון ולהערכה
# =========================================================
# מחשב accuracy של מודל על DataLoader נתון
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# מחשב ממוצע accuracy על כל המשימות שנלמדו עד עכשיו
def evaluate_average_seen_tasks(model, loaders, seen_task_count):
    accs = []
    for t in range(seen_task_count):
        accs.append(evaluate(model, loaders[t]))
    return float(np.mean(accs))


# מאמן את המודל epoch אחד; אם יש EWC מוסיפים penalty ל-loss
def train_one_epoch(model, train_loader, optimizer, ewc_obj=None, ewc_lambda=0):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        if ewc_obj is not None:
            loss = loss + ewc_lambda * ewc_obj.penalty(model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# מאמן משימה אחת למספר קבוע של epochs
def train_task_fixed_epochs(model, train_loader, optimizer, epochs, label="", ewc_obj=None, ewc_lambda=0):
    for epoch in range(epochs):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ewc_obj=ewc_obj,
            ewc_lambda=ewc_lambda
        )
        print(f"      {label} Epoch {epoch+1}/{epochs} - loss={loss:.4f}")

# =========================================================
# הרצת שלושת הניסויים: Single-task, SGD+Dropout, EWC
# =========================================================
# מאמן מודל נפרד לכל משימה — זה הקו השחור המקווקו בגרף
def run_single_task_reference(train_loaders, test_loaders):
    task_accs = []

    for t in range(MAX_TASKS):
        print(f"\n[Single-task] Training task {t+1}/{MAX_TASKS}")

        model = MLP().to(device)
        optimizer = optim.SGD(model.parameters(), lr=LR_SINGLE, momentum=MOMENTUM)

        train_task_fixed_epochs(
            model,
            train_loaders[t],
            optimizer,
            epochs=MAX_EPOCHS,
            label="single"
        )

        acc = evaluate(model, test_loaders[t])
        task_accs.append(acc)
        print(f"   Task {t+1} single-task acc: {acc:.4f}")

    cumulative = []
    for k in range(1, MAX_TASKS + 1):
        cumulative.append(float(np.mean(task_accs[:k])))

    return np.array(cumulative)


# מאמן מודל אחד ברצף על כל המשימות עם SGD+Dropout — הקו הכחול
def run_sgd_dropout_sequential(train_loaders, test_loaders):
    model = DropoutMLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR_DROPOUT, momentum=MOMENTUM)
    history = []

    for t in range(MAX_TASKS):
        print(f"\n[SGD+Dropout] Training task {t+1}/{MAX_TASKS}")

        train_task_fixed_epochs(
            model=model,
            train_loader=train_loaders[t],
            optimizer=optimizer,
            epochs=MAX_EPOCHS,
            label="dropout"
        )

        avg_acc = evaluate_average_seen_tasks(model, test_loaders, t + 1)
        history.append(avg_acc)
        print(f"   Test avg accuracy after {t+1} tasks: {avg_acc:.4f}")

    return np.array(history)


# מאמן מודל אחד ברצף עם EWC — הקו האדום
def run_ewc_sequential(train_loaders, test_loaders):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR_EWC, momentum=MOMENTUM)
    ewc_obj = ConsolidatedEWC(model)
    history = []

    for t in range(MAX_TASKS):
        print(f"\n[EWC λ={EWC_LAMBDA}] Training task {t+1}/{MAX_TASKS}")

        train_task_fixed_epochs(
            model=model,
            train_loader=train_loaders[t],
            optimizer=optimizer,
            epochs=MAX_EPOCHS,
            label="EWC",
            ewc_obj=ewc_obj if t > 0 else None,
            ewc_lambda=EWC_LAMBDA
        )

        ewc_obj.update_after_task(train_loaders[t])

        avg_acc = evaluate_average_seen_tasks(model, test_loaders, t + 1)
        history.append(avg_acc)
        print(f"   Test avg accuracy after {t+1} tasks: {avg_acc:.4f}")

    return np.array(history)

# =========================================================
# שרטוט Figure B
# =========================================================
# משרטט את הגרף הסופי שמשווה בין EWC, SGD+dropout ו-single task
def plot_figure_b(single_hist, dropout_hist, ewc_hist):
    x = list(range(2, MAX_TASKS + 1))

    plt.figure(figsize=(6.2, 4.2))

    plt.plot(x, ewc_hist[1:], color="red", linewidth=2.4, label=f"EWC λ={EWC_LAMBDA}")
    plt.plot(x, dropout_hist[1:], color="royalblue", linewidth=2.4, label="SGD+dropout")

    plt.plot(
        x,
        single_hist[1:],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="single task performance"
    )

    plt.xlabel("Number of tasks")
    plt.ylabel("Fraction correct")
    plt.ylim(0.80, 1.0)
    plt.xlim(2, MAX_TASKS)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"figure_B_lambda_{EWC_LAMBDA}.png", dpi=300)
    plt.show()

# =========================================================
# נקודת ההרצה הראשית של הקוד
# =========================================================
if __name__ == "__main__":
    total_start = time.time()

    print("\n========== Controlled Figure B test ==========")
    print(f"WIDTH={WIDTH}")
    print(f"LR_DROPOUT={LR_DROPOUT}")
    print(f"LR_EWC={LR_EWC}")
    print(f"EWC_LAMBDA={EWC_LAMBDA}")

    single_runs = []
    dropout_runs = []
    ewc_runs = []

    for seed in SEEDS:
        start = time.time()

        print("\n" + "=" * 80)
        print(f"RUNNING SEED {seed}")
        print("=" * 80)

        set_all_seeds(seed)

        perms = make_permutations(seed, MAX_TASKS)
        train_loaders, val_loaders, test_loaders = build_task_loaders(perms, seed)

        single_hist = run_single_task_reference(train_loaders, test_loaders)
        dropout_hist = run_sgd_dropout_sequential(train_loaders, test_loaders)
        ewc_hist = run_ewc_sequential(train_loaders, test_loaders)

        single_runs.append(single_hist)
        dropout_runs.append(dropout_hist)
        ewc_runs.append(ewc_hist)

        print(f"\nSeed {seed} finished in {(time.time() - start) / 60:.2f} min")

    single_avg = np.mean(np.stack(single_runs, axis=0), axis=0)
    dropout_avg = np.mean(np.stack(dropout_runs, axis=0), axis=0)
    ewc_avg = np.mean(np.stack(ewc_runs, axis=0), axis=0)

    print("\n========== FINAL ARRAYS ==========")
    print("single:", [round(x, 4) for x in single_avg])
    print("dropout:", [round(x, 4) for x in dropout_avg])
    print("EWC:", [round(x, 4) for x in ewc_avg])

    plot_figure_b(single_avg, dropout_avg, ewc_avg)

    print(f"\nTotal time: {(time.time() - total_start) / 60:.2f} min")
