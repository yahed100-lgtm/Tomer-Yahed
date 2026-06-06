# ייבוא ספריית PyTorch, המשמשת לעבודה עם טנזורים (tensors), חישובים מספריים, והרצת מודלים על CPU או GPU.
import torch

# ייבוא מודול הרשתות הנוירוניות של PyTorch, הכולל שכבות (layers) ומבנה בסיסי למודלים.
import torch.nn as nn

# ייבוא מודול האופטימיזציה (optimization), הכולל אלגוריתמים כמו SGD ו-Adam.
import torch.optim as optim

# ייבוא torchvision, שמספק גישה נוחה לדאטהסטים מוכנים כמו MNIST.
import torchvision

# ייבוא טרנספורמציות (transforms) לעיבוד מוקדם של התמונות לפני האימון.
import torchvision.transforms as transforms

# ייבוא DataLoader, שאחראי על טעינת הנתונים באצוות (batches) במהלך האימון והבדיקה.
from torch.utils.data import DataLoader

# ייבוא פונקציות שימושיות לרשתות נוירוניות, כמו loss functions ו-softmax.
import torch.nn.functional as F

# ייבוא matplotlib לצורך יצירת גרפים והצגת תוצאות הניסוי.
import matplotlib.pyplot as plt

# =====================================================================
# 1. תשתית והגדרות בסיסיות
# =====================================================================

# בחירת התקן החישוב (device): שימוש ב-GPU אם CUDA זמינה, אחרת שימוש ב-CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# קביעת seed קבוע כדי שהרצות חוזרות יהיו עקביות ככל האפשר מבחינת אקראיות.
torch.manual_seed(42)


# =====================================================================
# 2. הכנת הנתונים: Permuted MNIST
# =====================================================================

# פונקציה שמכינה את דאטהסט MNIST, עם אפשרות להחיל פרמוטציה (permutation) על הפיקסלים.
def get_permuted_mnist(permutation=None):

    # יצירת רשימת פעולות עיבוד: המרה לטנזור (ToTensor) ונרמול (Normalize) לפי ערכי MNIST המקובלים.
    transforms_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

    # אם התקבלה פרמוטציה, יוצרים גרסה חדשה של MNIST שבה סדר הפיקסלים שונה.
    if permutation is not None:

        # הפיכת התמונה לווקטור, סידור הפיקסלים לפי הפרמוטציה, והחזרה לצורת תמונה בגודל 1x28x28.
        transforms_list.append(transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28)))

    # איחוד כל פעולות העיבוד לרצף אחד באמצעות Compose.
    transform = transforms.Compose(transforms_list)

    # טעינת סט האימון (train set) של MNIST, כולל הורדה אוטומטית אם הדאטה עדיין לא קיים.
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # טעינת סט הבדיקה (test set) של MNIST עם אותן פעולות עיבוד.
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # החזרת DataLoader לאימון ולבדיקה; באימון יש ערבוב (shuffle), ובבדיקה אין צורך בערבוב.
    return DataLoader(train_ds, batch_size=256, shuffle=True), DataLoader(test_ds, batch_size=256, shuffle=False)


# הדפסת ההתקן שבו יתבצע החישוב: CPU או GPU.
print(f"Using device: {device}")

# הדפסת הודעה שמציינת את תחילת הכנת משימות Permuted MNIST.
print("Preparing Permuted MNIST tasks...")

# יצירת המשימה הראשונה: MNIST רגיל ללא שינוי בסדר הפיקסלים.
task1_train, task1_test = get_permuted_mnist(None)

# יצירת המשימה השנייה: MNIST עם פרמוטציה אקראית וקבועה של 784 הפיקסלים.
task2_train, task2_test = get_permuted_mnist(torch.randperm(784))

# יצירת המשימה השלישית: MNIST עם פרמוטציה אקראית אחרת של 784 הפיקסלים.
task3_train, task3_test = get_permuted_mnist(torch.randperm(784))

# איגוד שלוש המשימות לרשימה אחת, כדי לאמן את המודל עליהן ברצף.
tasks = [(task1_train, task1_test), (task2_train, task2_test), (task3_train, task3_test)]


# =====================================================================
# 3. ארכיטקטורת המודל
# =====================================================================

# הגדרת מודל מסוג MLP, כלומר רשת fully connected פשוטה לסיווג ספרות MNIST.
class MLP(nn.Module):

    # פונקציית האתחול של המודל, שבה מגדירים את שכבות הרשת.
    def __init__(self):

        # אתחול המחלקה הבסיסית nn.Module, כנדרש בכל מודל PyTorch.
        super().__init__()

        # הגדרת רצף השכבות של הרשת באמצעות Sequential.
        self.fc = nn.Sequential(

            # הפיכת התמונה הדו-ממדית לווקטור חד-ממדי בגודל 784.
            nn.Flatten(),

            # שכבה לינארית ראשונה: ממפה 784 פיקסלים ל-400 נוירונים, ולאחריה ReLU כאקטיבציה.
            nn.Linear(784, 400), nn.ReLU(),

            # שכבה לינארית שנייה: ממפה 400 נוירונים ל-400 נוירונים נוספים, ולאחריה ReLU.
            nn.Linear(400, 400), nn.ReLU(),

            # שכבת הפלט: ממפה את הייצוג הפנימי ל-10 מחלקות, אחת לכל ספרה.
            nn.Linear(400, 10)
        )

    # מעבר קדימה (forward pass): הקלט עובר דרך רצף השכבות ומוחזר כפלט גולמי (logits).
    def forward(self, x): return self.fc(x)


# =====================================================================
# 4. אלגוריתם Online EWC - מנגנון לצמצום שכחה קטסטרופלית
# =====================================================================

# מחלקה שמממשת Online EWC, כלומר גרסה שבה נשמר מידע מאוחד על חשיבות הפרמטרים ממשימות קודמות.
class OnlineEWC:

    # אתחול אובייקט EWC עם המודל הנוכחי ועם DataLoader של המשימה שעליה מחשבים את חשיבות הפרמטרים.
    def __init__(self, model, dataloader):

        # שמירת הפניה למודל שעליו יתבצע חישוב Fisher וה-penalty.
        self.model = model

        # שמירת עותק קפוא של פרמטרי המודל לאחר סיום משימה; זו נקודת הייחוס שאליה EWC מנסה להישאר קרוב.
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        # חישוב Fisher האלכסוני, המשמש כאומדן לחשיבות של כל פרמטר במודל.
        self.fisher = self._diag_fisher(dataloader)

    # פונקציה פנימית שמחשבת קירוב אלכסוני של Fisher Information Matrix.
    def _diag_fisher(self, dataloader):

        # יצירת מילון Fisher עם אפסים באותה צורה כמו כל פרמטר ניתן-למידה במודל.
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        # מעבר למצב הערכה (evaluation mode), כדי לחשב Fisher בלי התנהגות אימון כמו dropout.
        self.model.eval()

        # מעבר על כל האצוות בדאטה כדי לצבור מידע על רגישות הפרמטרים.
        for img, lbl in dataloader:

            # העברת התמונות והתוויות להתקן החישוב הנבחר.
            img, lbl = img.to(device), lbl.to(device)

            # איפוס גרדיאנטים קודמים לפני חישוב חדש.
            self.model.zero_grad()

            # חישוב פלט המודל עבור האצווה הנוכחית.
            out = self.model(img)

            # חישוב loss לפי התחזית של המודל עצמו, כדי לאמוד את יציבות הפרמטרים סביב הפתרון הנוכחי.
            loss = F.nll_loss(F.log_softmax(out, dim=1), out.argmax(dim=1))

            # חישוב גרדיאנטים של ה-loss ביחס לפרמטרי המודל.
            loss.backward()

            # מעבר על כל פרמטרי המודל לצורך צבירת ריבועי הגרדיאנטים.
            for n, p in self.model.named_parameters():

                # בדיקה שהפרמטר ניתן ללמידה ושקיים עבורו גרדיאנט.
                if p.requires_grad and p.grad is not None:

                    # הוספת ריבוע הגרדיאנט ל-Fisher; ערך גבוה מצביע על פרמטר חשוב יותר.
                    fisher[n].data += p.grad.data ** 2

        # מעבר על כל ערכי Fisher לצורך נרמול לפי מספר האצוות.
        for n in fisher:

            # חישוב ממוצע ריבועי הגרדיאנטים במקום סכום מצטבר.
            fisher[n].data /= len(dataloader)

        # החזרת מילון Fisher המחושב.
        return fisher

    # עדכון Fisher ונקודת הייחוס לאחר סיום משימה נוספת.
    def update(self, dataloader):

        # במקום לשמור משימות בנפרד, מחשבים ממוצע של הפישר כדי לשמור על כל הידע יחד
        # חישוב Fisher חדש עבור המשימה הנוכחית.
        new_fisher = self._diag_fisher(dataloader)

        # מעבר על כל הפרמטרים שעבורם קיים Fisher קודם.
        for n in self.fisher:

            # שילוב Fisher קודם עם Fisher חדש באמצעות ממוצע משוקלל פשוט.
            self.fisher[n].data = 0.5 * self.fisher[n].data + 0.5 * new_fisher[n].data

        # מעדכנים את נקודת העוגן של המשקולות לסיום המשימה הנוכחית
        # עדכון נקודת הייחוס של הפרמטרים למצב המודל לאחר סיום המשימה הנוכחית.
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    # חישוב קנס EWC, שמעניש שינוי גדול בפרמטרים החשובים למשימות קודמות.
    def penalty(self, model):

        # אתחול ערך הקנס ל-0.
        loss = 0

        # מעבר על פרמטרי המודל הנוכחי.
        for n, p in model.named_parameters():

            # בדיקה שהפרמטר קיים גם במילון הפרמטרים שנשמרו כנקודת ייחוס.
            if n in self.params:

                # הוספת קנס ריבועי: ככל שהפרמטר חשוב יותר לפי Fisher וככל שהוא התרחק יותר, הקנס גדל.
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()

        # החזרת סכום קנסות EWC עבור כל הפרמטרים.
        return loss


# =====================================================================
# 5. ניהול הניסוי המרכזי
# =====================================================================

# פונקציה שמריצה את הניסוי המלא, פעם כ-SGD רגיל ופעם עם Online EWC.
def run_experiment(use_ewc=False):

    # יצירת מופע חדש של מודל MLP והעברתו ל-device המתאים.
    model = MLP().to(device)

    # הגדרת האופטימייזר SGD עם קצב למידה (learning rate) של 0.1.
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # יצירת מבנה נתונים לשמירת הדיוק לאורך זמן עבור כל אחת משלוש המשימות.
    history = {0: [], 1: [], 2: []}

    # יצירת משתנה לאובייקט EWC; בתחילת הניסוי אין עדיין משימה קודמת לשמר.
    ewc_obj = None  # שימוש באובייקט אחד מאוחד במקום רשימה

    # מעבר סדרתי על שלוש המשימות, כפי שמקובל בניסוי Continual Learning.
    for t_idx, (train_loader, _) in enumerate(tasks):

        # הדפסת המשימה הנוכחית וסוג האימון: baseline רגיל או Online EWC.
        print(f"\nTraining Task {t_idx + 1} ({'Online EWC' if use_ewc else 'SGD Baseline'})...")

        # אימון כל משימה במשך 20 אפוקים.
        for epoch in range(20):

            # מעבר למצב אימון (training mode).
            model.train()

            # מעבר על כל האצוות של סט האימון במשימה הנוכחית.
            for batch_idx, (img, lbl) in enumerate(train_loader):

                # העברת התמונות והתוויות ל-device המתאים.
                img, lbl = img.to(device), lbl.to(device)

                # איפוס הגרדיאנטים לפני חישוב loss חדש.
                optimizer.zero_grad()

                # חישוב cross entropy loss עבור הסיווג במשימה הנוכחית.
                loss = F.cross_entropy(model(img), lbl)

                # אם עובדים עם EWC וכבר קיים ידע ממשימה קודמת, מוסיפים penalty ל-loss.
                if use_ewc and ewc_obj is not None:

                    # למבדה אגרסיבית שומרת על הידע המאוחד
                    # הוספת קנס EWC עם מקדם גדול, כדי להגביל שינוי בפרמטרים חשובים.
                    loss += 15000 * ewc_obj.penalty(model)

                # חישוב גרדיאנטים עבור ה-loss הכולל.
                loss.backward()

                # הגבלת גודל הגרדיאנטים (gradient clipping) כדי לשפר יציבות באימון.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                # עדכון פרמטרי המודל באמצעות SGD.
                optimizer.step()

            # הערכת הביצועים על כל המשימות
            # מעבר למצב הערכה לפני מדידת הדיוק.
            model.eval()

            # ביטול חישוב גרדיאנטים בזמן הערכה כדי לחסוך משאבים.
            with torch.no_grad():

                # בדיקת ביצועים על כל אחת משלוש המשימות.
                for eval_t_idx in range(3):

                    # אתחול מונים עבור מספר תחזיות נכונות ומספר דוגמאות כולל.
                    correct, total = 0, 0

                    # אם המשימה עדיין לא נלמדה, לא מודדים עליה דיוק בשלב הנוכחי.
                    if eval_t_idx > t_idx:

                        # הוספת NaN כדי שהגרף לא יציג ערך עבור משימה עתידית.
                        history[eval_t_idx].append(float('nan'))

                        # מעבר למשימה הבאה בהערכה.
                        continue

                    # מעבר על סט הבדיקה של המשימה הנבדקת.
                    for img, lbl in tasks[eval_t_idx][1]:

                        # העברת התמונות והתוויות ל-device המתאים.
                        img, lbl = img.to(device), lbl.to(device)

                        # חישוב תחזיות המודל וספירת מספר התחזיות הנכונות.
                        correct += (model(img).argmax(1) == lbl).sum().item()

                        # עדכון מספר הדוגמאות הכולל שנבדקו.
                        total += lbl.size(0)

                    # חישוב דיוק באחוזים ושמירתו בהיסטוריה של אותה משימה.
                    history[eval_t_idx].append(100 * correct / total)

            # הדפסת הדיוק של Task A לאחר כל אפוק, כדי לעקוב אחר שכחה קטסטרופלית.
            print(f"  Epoch {epoch + 1}/20 - Task A Acc: {history[0][-1]:.1f}%")

        # לאחר סיום משימה, אם משתמשים ב-EWC ולא מדובר במשימה האחרונה, מעדכנים את Fisher.
        if use_ewc and t_idx < 2:

            # הדפסת הודעה על עדכון Fisher Information Matrix.
            print("Updating Online Fisher Information Matrix...")

            # אם זהו סיום המשימה הראשונה, יוצרים את אובייקט EWC הראשון.
            if ewc_obj is None:

                # יצירת OnlineEWC על בסיס המודל לאחר שלמד את המשימה הראשונה.
                ewc_obj = OnlineEWC(model, train_loader)

            # אם כבר קיים אובייקט EWC, מעדכנים אותו על בסיס המשימה החדשה.
            else:

                # עדכון Fisher המאוחד ונקודת הייחוס של המשקולות.
                ewc_obj.update(train_loader)

    # החזרת היסטוריית הדיוק עבור כל המשימות לאורך כל שלבי האימון.
    return history


# =====================================================================
# 6. ציור הגרף
# =====================================================================

# בדיקה שהקוד הראשי ירוץ רק כאשר הקובץ מופעל ישירות, ולא כאשר הוא מיובא כמודול.
if __name__ == '__main__':

    # התחלת שלב ראשון: אימון baseline באמצעות SGD רגיל, ללא מנגנון למניעת שכחה.
    print("=== Phase 1: Training SGD Baseline ===")

    # הרצת הניסוי ללא EWC ושמירת היסטוריית הדיוקים.
    history_sgd = run_experiment(use_ewc=False)

    # התחלת שלב שני: אימון עם Online EWC.
    print("\n=== Phase 2: Training Online EWC ===")

    # הרצת הניסוי עם Online EWC ושמירת היסטוריית הדיוקים.
    history_ewc = run_experiment(use_ewc=True)

    # יצירת חלון גרפי עם שלושה גרפים אנכיים, אחד עבור כל task.
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    # הגדרת טווח האפוקים הכולל: 3 משימות כפול 20 אפוקים לכל משימה.
    epochs_range = range(1, 61)

    # הגדרת שמות המשימות שיופיעו לצד הגרפים.
    task_names = ['Task A', 'Task B', 'Task C']

    # מעבר על שלוש המשימות לצורך ציור גרף accuracy עבור כל אחת.
    for i in range(3):

        # בחירת תת-הגרף המתאים למשימה הנוכחית.
        ax = axes[i]

        # ציור עקומת הדיוק של SGD עבור המשימה הנוכחית.
        ax.plot(epochs_range, history_sgd[i], color='#4A70B0', label='SGD', linewidth=2)

        # ציור עקומת הדיוק של EWC עבור המשימה הנוכחית.
        ax.plot(epochs_range, history_ewc[i], color='#C84A48', label='EWC', linewidth=2)

        # הוספת קו אנכי שמסמן את המעבר מאימון Task A לאימון Task B.
        ax.axvline(x=20.5, color='gray', linestyle='--', alpha=0.7)

        # הוספת קו אנכי שמסמן את המעבר מאימון Task B לאימון Task C.
        ax.axvline(x=40.5, color='gray', linestyle='--', alpha=0.7)

        # הגדרת תווית ציר Y לפי שם המשימה.
        ax.set_ylabel(task_names[i], fontsize=12)

        # הגבלת טווח ציר Y כדי להתמקד באזור הדיוקים הגבוהים ובירידות עקב forgetting.
        ax.set_ylim(80, 102)

        # הסתרת המסגרת העליונה של הגרף למראה נקי יותר.
        ax.spines['top'].set_visible(False)

        # הסתרת המסגרת הימנית של הגרף למראה נקי יותר.
        ax.spines['right'].set_visible(False)

    # הוספת תווית שמראה שבחלק הראשון של ציר הזמן מתבצע אימון על Task A.
    axes[0].text(10.5, 105, 'train A', fontsize=12, ha='center')

    # הוספת תווית שמראה שבחלק השני של ציר הזמן מתבצע אימון על Task B.
    axes[0].text(30.5, 105, 'train B', fontsize=12, ha='center')

    # הוספת תווית שמראה שבחלק השלישי של ציר הזמן מתבצע אימון על Task C.
    axes[0].text(50.5, 105, 'train C', fontsize=12, ha='center')

    # הגדרת שם ציר X בגרף התחתון.
    axes[2].set_xlabel('Training time', fontsize=12)

    # הוספת מקרא שמבחין בין SGD לבין EWC.
    axes[0].legend(loc='lower left', bbox_to_anchor=(1.0, 0.5), frameon=False)

    # סידור אוטומטי של רכיבי הגרף כדי למנוע חפיפות.
    plt.tight_layout()

    # שמירת הגרף לקובץ PNG באיכות גבוהה.
    plt.savefig('figure_2A_online_perfect.png', dpi=300)

    # הדפסת הודעת סיום הכוללת את שם קובץ התמונה שנשמר.
    print("\nDone! Perfect Graph saved as 'figure_2A_online_perfect.png'")

    # הצגת הגרף על המסך.
    plt.show()
