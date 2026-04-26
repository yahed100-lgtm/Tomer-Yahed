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
# CONFIG — controlled next test from best result
# =========================================================
MAX_TASKS = 10

BATCH_SIZE = 256
WIDTH = 1500
MAX_EPOCHS = 100
PATIENCE = 5

LR_SINGLE = 0.001
LR_DROPOUT = 0.003
LR_EWC = 0.001
MOMENTUM = 0.9

# First controlled test: only change from best code
EWC_LAMBDA = 12000

VAL_FRACTION = 0.1
SEEDS = [0]

NUM_WORKERS = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================================
# DATA
# =========================================================
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


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_permutations(seed, max_tasks=10):
    g = torch.Generator()
    g.manual_seed(seed)

    perms = [torch.arange(784)]
    for _ in range(max_tasks - 1):
        perms.append(torch.randperm(784, generator=g))

    return perms


def apply_permutation(x, perm):
    flat = x.view(x.size(0), -1)
    flat = flat[:, perm]
    return flat.view(-1, 1, 28, 28)


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
# MODELS
# =========================================================
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
# EWC — same version as best-looking code
# =========================================================
class ConsolidatedEWC:
    def __init__(self, model):
        self.model = model
        self.fisher = None
        self.star_params = None

    def _snapshot_params(self):
        return {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

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

    def update_after_task(self, dataloader):
        current_fisher = self._diag_fisher(dataloader)
        current_params = self._snapshot_params()

        if self.fisher is None:
            self.fisher = current_fisher
            self.star_params = current_params
        else:
            for n in self.fisher:
                self.fisher[n] = self.fisher[n] + current_fisher[n]

            # keep same behavior as best result
            self.star_params = current_params

    def penalty(self, model):
        if self.fisher is None or self.star_params is None:
            return torch.tensor(0.0, device=device)

        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()

        return loss

# =========================================================
# HELPERS
# =========================================================
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


def evaluate_average_seen_tasks(model, loaders, seen_task_count):
    accs = []
    for t in range(seen_task_count):
        accs.append(evaluate(model, loaders[t]))
    return float(np.mean(accs))


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
# EXPERIMENTS
# =========================================================
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
# PLOT
# =========================================================
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
# MAIN
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