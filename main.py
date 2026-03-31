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
torch.manual_seed(42)  # הגדרת Seed קבוע היא חובה בשחזור מדעי כדי להבטיח שהגרף ייראה בדיוק אותו דבר בכל הרצה.


# =====================================================================
# 2. הכנת הנתונים: Permuted MNIST (הטריק של המאמר המקורי)
# =====================================================================
# למה Permuted? במאמר המקורי, כדי לייצר משימות חדשות מבלי לשנות את מבנה הרשת,
# החוקרים פשוט "ערבבו" את סדר הפיקסלים של התמונות. עבור הרשת, זה נראה כמו משימה חדשה לגמרי.
def get_permuted_mnist(permutation=None):
    transforms_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

    # אם קיבלנו וקטור ערבוב, נשנה את מיקום הפיקסלים בתמונה
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28)))

    transform = transforms.Compose(transforms_list)
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(train_ds, batch_size=256, shuffle=True), DataLoader(test_ds, batch_size=256, shuffle=False)


print("Preparing Permuted MNIST tasks...")
task1_train, task1_test = get_permuted_mnist(None)  # משימה A: התמונות המקוריות ללא ערבוב
task2_train, task2_test = get_permuted_mnist(torch.randperm(784))  # משימה B: ערבוב אקראי של כל 784 הפיקסלים
task3_train, task3_test = get_permuted_mnist(torch.randperm(784))  # משימה C: ערבוב אקראי שונה
tasks = [(task1_train, task1_test), (task2_train, task2_test), (task3_train, task3_test)]


# =====================================================================
# 3. ארכיטקטורת המודל (MLP)
# =====================================================================
# שחזור מדויק של הרשת מהמאמר: רשת מרובת שכבות (MLP) עם שתי שכבות נסתרות של 400 נוירונים.
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 10)  # 10 מחלקות בסוף (עבור 10 הספרות)
        )

    def forward(self, x): return self.fc(x)


# =====================================================================
# 4. הלב של הפרויקט: אלגוריתם EWC
# =====================================================================
class EWC:
    def __init__(self, model, dataloader):
        self.model = model
        # 1. שמירת עוגן: שומרים את המשקולות האופטימליות של המשימה שכבר למדנו.
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        # 2. חישוב פישר: מוצאים אילו משקולות היו קריטיות להצלחה של המשימה הקודמת.
        self.fisher = self._diag_fisher(dataloader)

    def _diag_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)
            self.model.zero_grad()
            out = self.model(img)
            # לפי המאמר, הפישר האמפירי מחושב מתוך נגזרת הלוג-הסתברות
            loss = F.nll_loss(F.log_softmax(out, dim=1), out.argmax(dim=1))
            loss.backward()

            # צבירת ריבוע הגרדיאנטים (זה הקירוב למטריצת פישר)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n].data += p.grad.data ** 2

        # מיצוע על פני כל הנתונים
        for n in fisher:
            fisher[n].data /= len(dataloader)
        return fisher

    def penalty(self, model):
        # חישוב ה"קנס": אם משקולת חשובה (פישר גבוה) זזה הרבה מהעוגן שלה, ה-Loss יקפוץ לשמיים.
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss


# =====================================================================
# 5. ניהול הניסוי המרכזי (Baseline vs. EWC)
# =====================================================================
def run_experiment(use_ewc=False):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    history = []
    ewc_list = []

    for t_idx, (train_loader, _) in enumerate(tasks):
        print(f"Training Task {t_idx + 1} ({'EWC Protection' if use_ewc else 'SGD Baseline'})...")
        # 10 איטרציות (Epochs) מספיקות כדי לאפשר ל-SGD לדרוס את הידע הישן
        for epoch in range(10):
            model.train()
            for img, lbl in train_loader:
                img, lbl = img.to(device), lbl.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(img), lbl)

                # הוספת ה"קנס" של EWC לפונקציית ה-Loss הרגילה
                if use_ewc:
                    for ewc in ewc_list:
                        loss += 2000 * ewc.penalty(model)  # Lambda = 2000 לאיזון הגמישות והיציבות

                loss.backward()

                # Gradient Clipping: מונע מהגרדיאנטים להתפוצץ (בעיה נפוצה בשילוב עונשים גבוהים)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()

            # הערכת הביצועים *רק* על משימה A כדי לעקוב אחרי השכחה
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for img, lbl in tasks[0][1]:
                    img, lbl = img.to(device), lbl.to(device)
                    correct += (model(img).argmax(1) == lbl).sum().item()
                    total += lbl.size(0)
            history.append(100 * correct / total)

        # בסיום למידת המשימה, מחשבים את מטריצת הפישר שלה כדי להגן עליה במשימה הבאה
        if use_ewc and t_idx < 2:
            print("Calculating Fisher Information Matrix for task protection...")
            ewc_list.append(EWC(model, train_loader))

    return history


# =====================================================================
# 6. הרצה וציור הגרף (קוסמטיקה זהה למאמר המקורי)
# =====================================================================
if __name__ == '__main__':
    print("=== Phase 1: SGD Baseline (Catastrophic Forgetting) ===")
    history_sgd = run_experiment(use_ewc=False)

    print("\n=== Phase 2: EWC (Memory Retention) ===")
    history_ewc = run_experiment(use_ewc=True)

    plt.figure(figsize=(10, 5))  # פרופורציות דומות למאמר

    epochs_range = range(1, 31)

    # הסרתי את ה-markers (הנקודות העבות) כדי לקבל קו חלק וזורם כמו בפרסום אקדמי
    # שימוש בצבעים הקלאסיים של DeepMind
    plt.plot(epochs_range, history_sgd, color='#4A70B0', label='SGD', linewidth=2.5)
    plt.plot(epochs_range, history_ewc, color='#C84A48', label='EWC', linewidth=2.5)

    # סימון גבולות המשימות בקווי מקווקוים עדינים
    plt.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=20.5, color='gray', linestyle='--', alpha=0.7)

    # מיקום הטקסט בזהירות שלא יחתך
    plt.text(5.5, 98, 'train A', fontsize=12, ha='center', color='black')
    plt.text(15.5, 98, 'train B', fontsize=12, ha='center', color='black')
    plt.text(25.5, 98, 'train C', fontsize=12, ha='center', color='black')

    # הגבלת ציר Y מ-80 ל-100 בדיוק כמו במאמר המקורי (0.8 עד 1.0)
    plt.ylim(80, 100)

    plt.ylabel('Fraction correct (Task A %)', fontsize=11)
    plt.xlabel('Training time (Epochs)', fontsize=11)

    # הסרת הרשת (Grid) כדי לתת מראה נקי כמו במאמר המקורי
    plt.grid(False)
    plt.legend(loc='lower left', frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig('figure_1A_publication_ready.png', dpi=300)  # איכות גבוהה
    plt.show()