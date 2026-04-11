import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =====================================================================
# 1. תשתית והגדרות בסיס
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# =====================================================================
# 2. הכנת הנתונים: Permuted MNIST
# =====================================================================
def get_permuted_mnist(permutation=None):
    transforms_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28)))

    transform = transforms.Compose(transforms_list)
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(train_ds, batch_size=256, shuffle=True), DataLoader(test_ds, batch_size=256, shuffle=False)


print(f"Using device: {device}")
print("Preparing Permuted MNIST tasks...")
task1_train, task1_test = get_permuted_mnist(None)
task2_train, task2_test = get_permuted_mnist(torch.randperm(784))
task3_train, task3_test = get_permuted_mnist(torch.randperm(784))
tasks = [(task1_train, task1_test), (task2_train, task2_test), (task3_train, task3_test)]


# =====================================================================
# 3. ארכיטקטורת המודל
# =====================================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 10)
        )

    def forward(self, x): return self.fc(x)


# =====================================================================
# 4. אלגוריתם Online EWC - הגרסה המתקדמת למניעת התנגשויות
# =====================================================================
class OnlineEWC:
    def __init__(self, model, dataloader):
        self.model = model
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._diag_fisher(dataloader)

    def _diag_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)
            self.model.zero_grad()
            out = self.model(img)
            loss = F.nll_loss(F.log_softmax(out, dim=1), out.argmax(dim=1))
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n].data += p.grad.data ** 2

        for n in fisher:
            fisher[n].data /= len(dataloader)
        return fisher

    def update(self, dataloader):
        # במקום לשמור משימות בנפרד, מחשבים ממוצע של הפישר כדי לשמור על כל הידע יחד
        new_fisher = self._diag_fisher(dataloader)
        for n in self.fisher:
            self.fisher[n].data = 0.5 * self.fisher[n].data + 0.5 * new_fisher[n].data
        # מעדכנים את נקודת העוגן של המשקולות לסיום המשימה הנוכחית
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss


# =====================================================================
# 5. ניהול הניסוי המרכזי
# =====================================================================
def run_experiment(use_ewc=False):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    history = {0: [], 1: [], 2: []}
    ewc_obj = None  # שימוש באובייקט אחד מאוחד במקום רשימה

    for t_idx, (train_loader, _) in enumerate(tasks):
        print(f"\nTraining Task {t_idx + 1} ({'Online EWC' if use_ewc else 'SGD Baseline'})...")
        for epoch in range(20):
            model.train()
            for batch_idx, (img, lbl) in enumerate(train_loader):
                img, lbl = img.to(device), lbl.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(img), lbl)

                if use_ewc and ewc_obj is not None:
                    # למבדה אגרסיבית שומרת על הידע המאוחד
                    loss += 15000 * ewc_obj.penalty(model)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            # הערכת הביצועים על כל המשימות
            model.eval()
            with torch.no_grad():
                for eval_t_idx in range(3):
                    correct, total = 0, 0
                    if eval_t_idx > t_idx:
                        history[eval_t_idx].append(float('nan'))
                        continue

                    for img, lbl in tasks[eval_t_idx][1]:
                        img, lbl = img.to(device), lbl.to(device)
                        correct += (model(img).argmax(1) == lbl).sum().item()
                        total += lbl.size(0)
                    history[eval_t_idx].append(100 * correct / total)

            print(f"  Epoch {epoch + 1}/20 - Task A Acc: {history[0][-1]:.1f}%")

        if use_ewc and t_idx < 2:
            print("Updating Online Fisher Information Matrix...")
            if ewc_obj is None:
                ewc_obj = OnlineEWC(model, train_loader)
            else:
                ewc_obj.update(train_loader)

    return history


# =====================================================================
# 6. ציור הגרף
# =====================================================================
if __name__ == '__main__':
    print("=== Phase 1: Training SGD Baseline ===")
    history_sgd = run_experiment(use_ewc=False)

    print("\n=== Phase 2: Training Online EWC ===")
    history_ewc = run_experiment(use_ewc=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    epochs_range = range(1, 61)
    task_names = ['Task A', 'Task B', 'Task C']

    for i in range(3):
        ax = axes[i]
        ax.plot(epochs_range, history_sgd[i], color='#4A70B0', label='SGD', linewidth=2)
        ax.plot(epochs_range, history_ewc[i], color='#C84A48', label='EWC', linewidth=2)

        ax.axvline(x=20.5, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=40.5, color='gray', linestyle='--', alpha=0.7)

        ax.set_ylabel(task_names[i], fontsize=12)
        ax.set_ylim(80, 102)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].text(10.5, 105, 'train A', fontsize=12, ha='center')
    axes[0].text(30.5, 105, 'train B', fontsize=12, ha='center')
    axes[0].text(50.5, 105, 'train C', fontsize=12, ha='center')

    axes[2].set_xlabel('Training time', fontsize=12)
    axes[0].legend(loc='lower left', bbox_to_anchor=(1.0, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig('figure_2A_online_perfect.png', dpi=300)
    print("\nDone! Perfect Graph saved as 'figure_2A_online_perfect.png'")
    plt.show()