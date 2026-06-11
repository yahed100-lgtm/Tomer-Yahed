# =========================================================
# שחזור Figure B ממאמר EWC על Permuted MNIST
# מטרת הקוד היא לשחזר גרף בסגנון Figure B ממאמר EWC, שבו בודקים שכחה קטסטרופלית (catastrophic forgetting).
# הקוד מריץ שלושה ניסויים:
# כאן נבנות שלוש שיטות השוואה כדי לראות איך המודל מתמודד עם למידה רציפה (continual learning).
# 1. Single-task reference — מודל נפרד לכל משימה, קו שחור מקווקו
# זוהי נקודת ייחוס: כל task מקבל מודל משלו, ולכן אין שכחה בין משימות.
# 2. SGD + Dropout — מודל אחד שנלמד ברצף, קו כחול
# זהו baseline שבו אותו מודל לומד משימה אחרי משימה, ולכן ניתן לראות כמה הוא שוכח משימות קודמות.
# 3. EWC — מודל אחד שנלמד ברצף עם ענישה על שינוי משקולות חשובות, קו אדום
# זהו הניסוי המרכזי: EWC מוסיף penalty כדי למנוע שינוי גדול במשקולות החשובות למשימות קודמות.
# =========================================================

# ייבוא מודול time למדידת זמני ריצה של כל seed ושל כל הניסוי.
import time

# ייבוא copy לצורך העתקות עמוקות או רדודות של אובייקטים; בקוד הנוכחי הוא לא נמצא בשימוש פעיל.
import copy

# ייבוא NumPy לעבודה עם מערכים, חישוב ממוצעים, וארגון תוצאות הניסויים.
import numpy as np

# ייבוא PyTorch, הספרייה המרכזית לעבודה עם tensors, מודלים, אימון והרצה על CPU/GPU.
import torch

# ייבוא מודול הרשתות הנוירוניות של PyTorch, כולל שכבות ומחלקת הבסיס Module.
import torch.nn as nn

# ייבוא פונקציות שימושיות כמו cross_entropy, שמשמשת כאן כ-loss function.
import torch.nn.functional as F

# ייבוא מודול האופטימיזציה, שמספק את SGD עם momentum.
import torch.optim as optim

# ייבוא כלים לטעינת דאטה: DataLoader לאצוות, TensorDataset לדאטה מטנזורים, ו-random_split לחלוקת train/validation.
from torch.utils.data import DataLoader, TensorDataset, random_split

# ייבוא torchvision כדי לטעון את דאטהסט MNIST בצורה נוחה.
import torchvision

# ייבוא matplotlib לציור הגרף הסופי של Figure B.
import matplotlib.pyplot as plt

# =========================================================
# הגדרות הניסוי — הערכים שנבחרו לפי הריצה שנתנה את הגרף הכי טוב
# כאן מוגדרים ההיפר-פרמטרים (hyperparameters) המרכזיים של הניסוי.
# =========================================================

# מספר המשימות הרציפות בניסוי Permuted MNIST; כל task הוא גרסה אחרת של MNIST.
MAX_TASKS = 10          # מספר המשימות הרציפות ב-Permuted MNIST

# גודל האצווה (batch size) בכל צעד אימון.
BATCH_SIZE = 256       # גודל batch באימון

# מספר הנוירונים בכל שכבה חבויה של רשת ה-MLP.
WIDTH = 1500           # מספר נוירונים בכל שכבה חבויה

# מספר ה-epochs שבהם מאמנים כל task.
MAX_EPOCHS = 100      # מספר epochs לכל משימה

# פרמטר patience נשאר מניסויים קודמים, אך בגרסה הזו אין early stopping ולכן הוא לא בשימוש.
PATIENCE = 5          # לא בשימוש בגרסה הזו, נשאר מהניסויים הקודמים

# קצב הלמידה (learning rate) עבור מודלים שאומנו בנפרד לכל task.
LR_SINGLE = 0.001    # learning rate לאימון single-task

# קצב הלמידה עבור baseline של SGD עם Dropout.
LR_DROPOUT = 0.003   # learning rate ל-baseline עם dropout

# קצב הלמידה עבור המודל שנלמד עם EWC.
LR_EWC = 0.001       # learning rate לאימון עם EWC

# momentum משמש את SGD כדי להחליק את כיוון העדכון ולייצב את האימון.
MOMENTUM = 0.9       # momentum עבור SGD

# מקדם הענישה של EWC — הערך שנבחר אחרי ניסויים, נתן את התוצאה הטובה ביותר
# Lambda קובע כמה חזק ה-EWC יתנגד לשינוי בפרמטרים חשובים ממשימות קודמות.
EWC_LAMBDA = 12000  # חוזק הענישה של EWC

# אחוז הדאטה שיופרד מסט האימון וישמש כ-validation set.
VAL_FRACTION = 0.1   # אחוז הדאטה שמופרד ל-validation

SEEDS = [0, 1, 2]      # שלושה seeds כדי לקבל ממוצע אמין יותר

# מספר תהליכי worker לטעינת דאטה; 0 פשוט ונוח במיוחד בסביבות כמו Windows או Kaggle.
NUM_WORKERS = 0       # מספר workers ל-DataLoader; 0 מתאים ל-Windows/Kaggle פשוט

# בחירת התקן החישוב: GPU אם CUDA זמינה, אחרת CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# הדפסת ההתקן שבו הקוד ירוץ.
print("Using device:", device)

# בדיקה האם קיימת CUDA, כלומר האם יש GPU זמין ל-PyTorch.
if torch.cuda.is_available():

    # אם יש GPU, מדפיסים את שם הכרטיס הגרפי.
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================================
# טעינת הנתונים ובניית משימות Permuted MNIST
# בחלק הזה נטען MNIST פעם אחת, ולאחר מכן ניצור ממנו כמה tasks באמצעות permutation שונה לפיקסלים.
# =========================================================

# טוען את MNIST, מנרמל את הפיקסלים, ומחזיר train/test מלאים
# הפונקציה מחזירה את התמונות והתוויות כ-tensors מוכנים לשימוש.
def load_base_mnist():

    # טעינת סט האימון המקורי של MNIST.
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)

    # טעינת סט הבדיקה המקורי של MNIST.
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    # המרת תמונות האימון ל-float, חלוקה ב-255 כדי לקבל ערכים בין 0 ל-1, והוספת ממד channel.
    x_train = train_ds.data.float().div(255.0).unsqueeze(1)

    # המרת תמונות הבדיקה באותו אופן: float, נרמול ל-0 עד 1, והוספת channel.
    x_test = test_ds.data.float().div(255.0).unsqueeze(1)

    # נרמול תמונות האימון לפי mean ו-std המקובלים של MNIST.
    x_train = (x_train - 0.1307) / 0.3081

    # נרמול תמונות הבדיקה באותו אופן כדי לשמור על עקביות בין train ל-test.
    x_test = (x_test - 0.1307) / 0.3081

    # שליפת תוויות האימון והמרתן לטיפוס long, כנדרש ב-cross_entropy.
    y_train = train_ds.targets.long()

    # שליפת תוויות הבדיקה והמרתן לטיפוס long.
    y_test = test_ds.targets.long()

    # החזרת כל הטנזורים הבסיסיים של MNIST.
    return x_train, y_train, x_test, y_test


# טעינת MNIST פעם אחת ושמירתו כבסיס לכל משימות ה-Permuted MNIST.
BASE_X_TRAIN, BASE_Y_TRAIN, BASE_X_TEST, BASE_Y_TEST = load_base_mnist()


# קובע seed כדי שהריצה תהיה כמה שיותר שחזורית
# הפונקציה מקבעת את מקורות האקראיות המרכזיים כדי שהתוצאות יהיו reproducible ככל האפשר.
def set_all_seeds(seed):

    # קביעת seed עבור NumPy.
    np.random.seed(seed)

    # קביעת seed עבור PyTorch על CPU.
    torch.manual_seed(seed)

    # אם יש GPU, מקבעים גם את ה-seed של CUDA.
    if torch.cuda.is_available():

        # קביעת seed לכל התקני CUDA הזמינים.
        torch.cuda.manual_seed_all(seed)


# יוצר 10 משימות: הראשונה MNIST רגיל, והשאר עם פרמוטציה אקראית קבועה לפיקסלים
# כל permutation מגדירה task חדש שבו סדר הפיקסלים בתמונה השתנה, אבל התווית נשארת אותה ספרה.
def make_permutations(seed, max_tasks=10):

    # יצירת generator מקומי של PyTorch כדי לשלוט באקראיות של הפרמוטציות.
    g = torch.Generator()

    # קביעת seed ל-generator כדי שאותן פרמוטציות ייווצרו בכל ריצה עם אותו seed.
    g.manual_seed(seed)

    # המשימה הראשונה משתמשת בסדר הפיקסלים המקורי, כלומר ללא permutation.
    perms = [torch.arange(784)]

    # יצירת פרמוטציות אקראיות עבור שאר המשימות.
    for _ in range(max_tasks - 1):

        # הוספת permutation אקראית של 784 הפיקסלים עבור task חדש.
        perms.append(torch.randperm(784, generator=g))

    # החזרת רשימת הפרמוטציות עבור כל המשימות.
    return perms


# מחיל את הפרמוטציה על תמונות MNIST: הופך לוקטור, מסדר פיקסלים מחדש, ומחזיר לצורת 28x28
# הפונקציה מקבלת tensor של תמונות ומחזירה את אותן תמונות אחרי שינוי סדר הפיקסלים.
def apply_permutation(x, perm):

    # הפיכת כל תמונה לווקטור באורך 784 פיקסלים.
    flat = x.view(x.size(0), -1)

    # סידור מחדש של הפיקסלים לפי ה-permutation שנבחרה.
    flat = flat[:, perm]

    # החזרת התמונות לצורה המקורית: מספר דוגמאות, channel אחד, גובה 28 ורוחב 28.
    return flat.view(-1, 1, 28, 28)


# בונה DataLoaders לכל משימה: train, validation, test
# הפונקציה מייצרת loaders נפרדים לכל task, כדי שהאימון וההערכה יהיו מסודרים.
def build_task_loaders(perms, seed):

    # רשימה שתכיל את DataLoader-ים של האימון עבור כל task.
    train_loaders = []

    # רשימה שתכיל את DataLoader-ים של ה-validation עבור כל task.
    val_loaders = []

    # רשימה שתכיל את DataLoader-ים של הבדיקה עבור כל task.
    test_loaders = []

    # יצירת generator עבור חלוקת train/validation בצורה שחזורית.
    split_gen = torch.Generator()

    # קביעת seed שונה מעט עבור פעולת הפיצול, כדי לא להתנגש ישירות עם פרמוטציות הפיקסלים.
    split_gen.manual_seed(seed + 12345)

    # מעבר על כל המשימות הרציפות שהוגדרו.
    for task_idx in range(MAX_TASKS):

        # יצירת סט אימון עבור task מסוים על ידי החלת הפרמוטציה המתאימה.
        x_train = apply_permutation(BASE_X_TRAIN, perms[task_idx])

        # יצירת סט בדיקה עבור אותו task עם אותה פרמוטציה בדיוק.
        x_test = apply_permutation(BASE_X_TEST, perms[task_idx])

        # יצירת TensorDataset עבור סט האימון, עם התמונות המופרמטות והתוויות המקוריות.
        full_train_ds = TensorDataset(x_train, BASE_Y_TRAIN)

        # יצירת TensorDataset עבור סט הבדיקה.
        test_ds = TensorDataset(x_test, BASE_Y_TEST)

        # חישוב גודל סט ה-validation לפי האחוז שהוגדר ב-VAL_FRACTION.
        val_size = int(len(full_train_ds) * VAL_FRACTION)

        # חישוב גודל סט האימון לאחר שמוציאים ממנו את ה-validation.
        train_size = len(full_train_ds) - val_size

        # חלוקת סט האימון המלא ל-train ו-validation.
        train_ds, val_ds = random_split(

            # הדאטהסט המלא שממנו מבצעים את החלוקה.
            full_train_ds,

            # גדלי הפיצול: train ואז validation.
            [train_size, val_size],

            # שימוש ב-generator כדי שהחלוקה תהיה שחזורית.
            generator=split_gen
        )

        # יצירת DataLoader עבור סט האימון של task מסוים.
        train_loaders.append(DataLoader(

            # הדאטהסט שישמש לאימון.
            train_ds,

            # גודל ה-batch באימון.
            batch_size=BATCH_SIZE,

            # ערבוב הדוגמאות בכל epoch כדי לשפר את תהליך האימון.
            shuffle=True,

            # מספר worker-ים לטעינת הדאטה.
            num_workers=NUM_WORKERS,

            # שימוש ב-pin_memory כאשר עובדים עם CUDA כדי לשפר העברת נתונים ל-GPU.
            pin_memory=torch.cuda.is_available()
        ))

        # יצירת DataLoader עבור סט ה-validation.
        val_loaders.append(DataLoader(

            # הדאטהסט שישמש ל-validation.
            val_ds,

            # גודל ה-batch בזמן validation.
            batch_size=BATCH_SIZE,

            # אין צורך לערבב בזמן validation כי לא מתבצע אימון.
            shuffle=False,

            # מספר worker-ים לטעינת הדאטה.
            num_workers=NUM_WORKERS,

            # שימוש ב-pin_memory כאשר CUDA זמינה.
            pin_memory=torch.cuda.is_available()
        ))

        # יצירת DataLoader עבור סט הבדיקה של task מסוים.
        test_loaders.append(DataLoader(

            # הדאטהסט שישמש להערכה סופית.
            test_ds,

            # גודל ה-batch בזמן בדיקה.
            batch_size=BATCH_SIZE,

            # אין צורך לערבב בזמן בדיקה כי רק מחשבים accuracy.
            shuffle=False,

            # מספר worker-ים לטעינת הדאטה.
            num_workers=NUM_WORKERS,

            # שימוש ב-pin_memory אם עובדים עם GPU.
            pin_memory=torch.cuda.is_available()
        ))

    # החזרת שלוש רשימות loaders: אימון, validation ובדיקה.
    return train_loaders, val_loaders, test_loaders

# =========================================================
# הגדרת המודלים
# כאן מוגדרות שתי ארכיטקטורות: MLP רגיל ו-MLP עם Dropout.
# =========================================================

# רשת MLP רגילה — משמשת ל-Single Task ול-EWC
# זהו מודל fully connected פשוט, כמו בניסויי Permuted MNIST רבים.
class MLP(nn.Module):

    # אתחול המודל עם רוחב שכבות חבויות שנקבע לפי WIDTH.
    def __init__(self, width=WIDTH):

        # אתחול מחלקת הבסיס nn.Module.
        super().__init__()

        # הגדרת רצף השכבות של המודל.
        self.net = nn.Sequential(

            # הפיכת תמונת 28x28 לווקטור באורך 784.
            nn.Flatten(),

            # שכבה לינארית ראשונה: מ-784 פיקסלים ל-width נוירונים.
            nn.Linear(784, width),

            # פונקציית אקטיבציה ReLU להוספת אי-לינאריות.
            nn.ReLU(),

            # שכבה לינארית שנייה: מ-width ל-width.
            nn.Linear(width, width),

            # אקטיבציה נוספת מסוג ReLU.
            nn.ReLU(),

            # שכבת פלט שמחזירה 10 ערכים, אחד עבור כל ספרה ב-MNIST.
            nn.Linear(width, 10)
        )

    # הגדרת מעבר קדימה (forward pass) של המודל.
    def forward(self, x):

        # העברת הקלט דרך רצף השכבות והחזרת logits.
        return self.net(x)


# רשת MLP עם Dropout — משמשת כ-baseline של SGD+dropout
# Dropout משמש כ-regularization, אך אינו מיועד ספציפית למנוע שכחה קטסטרופלית.
class DropoutMLP(nn.Module):

    # אתחול מודל ה-DropoutMLP עם אותו רוחב שכבות.
    def __init__(self, width=WIDTH):

        # אתחול מחלקת הבסיס nn.Module.
        super().__init__()

        # הגדרת רצף השכבות עם Dropout בין חלק מהשכבות.
        self.net = nn.Sequential(

            # הפיכת התמונה לווקטור שטוח באורך 784.
            nn.Flatten(),

            # Dropout על הקלט, שמאפס באקראי 20% מהערכים בזמן אימון.
            nn.Dropout(0.2),

            # שכבה לינארית ראשונה.
            nn.Linear(784, width),

            # אקטיבציית ReLU.
            nn.ReLU(),

            # Dropout חזק יותר בשכבה החבויה, שמאפס 50% מהנוירונים בזמן אימון.
            nn.Dropout(0.5),

            # שכבה לינארית שנייה.
            nn.Linear(width, width),

            # אקטיבציית ReLU נוספת.
            nn.ReLU(),

            # Dropout נוסף לפני שכבת הפלט.
            nn.Dropout(0.5),

            # שכבת פלט ל-10 מחלקות.
            nn.Linear(width, 10)
        )

    # הגדרת מעבר קדימה של המודל.
    def forward(self, x):

        # העברת הקלט דרך הרשת והחזרת logits.
        return self.net(x)

# =========================================================
# מימוש EWC — אותו מימוש שנתן את התוצאה הוויזואלית הכי טובה
# EWC שומר מידע על חשיבות פרמטרים ומעניש שינוי שלהם במשימות הבאות.
# =========================================================

# מחלקה שמחשבת ושומרת Fisher + פרמטרים קודמים לצורך ענישת EWC
# כאן ה-Fisher מצטבר בין משימות, ולכן מדובר במימוש מאוחד (consolidated) של EWC.
class ConsolidatedEWC:

    # אתחול אובייקט EWC עבור מודל נתון.
    def __init__(self, model):

        # שמירת הפניה למודל שעליו מחשבים Fisher ו-penalty.
        self.model = model

        # בתחילת הריצה עדיין אין Fisher, כי עוד לא הסתיימה אף משימה.
        self.fisher = None

        # בתחילת הריצה עדיין אין נקודת עוגן של פרמטרים קודמים.
        self.star_params = None

    # שומר עותק של הפרמטרים הנוכחיים של הרשת אחרי סיום משימה
    # snapshot משמש כנקודת ייחוס: לאן הפרמטרים צריכים להישאר קרובים.
    def _snapshot_params(self):

        # החזרת מילון של שמות פרמטרים מול עותק מנותק שלהם מהגרף החישובי.
        return {

            # n הוא שם הפרמטר, ו-p הוא הטנזור של הפרמטר עצמו.
            n: p.clone().detach()

            # מעבר על כל הפרמטרים עם שמותיהם במודל.
            for n, p in self.model.named_parameters()

            # שמירה רק של פרמטרים שניתנים ללמידה.
            if p.requires_grad
        }

    # מחשב קירוב אלכסוני למטריצת Fisher לפי הגרדיאנטים של loss על הדאטה
    # Fisher משמש כאומדן לחשיבות של כל משקל עבור המשימה שנלמדה.
    def _diag_fisher(self, dataloader):

        # יצירת מילון Fisher עם אפסים באותה צורה כמו כל פרמטר ניתן-למידה.
        fisher = {

            # לכל פרמטר יוצרים tensor אפסים באותה צורה ועל אותו device.
            n: torch.zeros_like(p, device=device)

            # מעבר על כל הפרמטרים של המודל.
            for n, p in self.model.named_parameters()

            # חישוב Fisher רק עבור פרמטרים שניתן לעדכן באימון.
            if p.requires_grad
        }

        # מעבר למצב הערכה כדי שלא תהיה התנהגות אימון כמו Dropout.
        self.model.eval()

        # מונה את מספר ה-batches שעליהם חושב Fisher.
        batch_count = 0

        # מעבר על כל האצוות ב-DataLoader.
        for images, labels in dataloader:

            # העברת התמונות ל-device; non_blocking יכול לשפר ביצועים כאשר pin_memory פעיל.
            images = images.to(device, non_blocking=True)

            # העברת התוויות לאותו device.
            labels = labels.to(device, non_blocking=True)

            # איפוס גרדיאנטים קודמים לפני חישוב חדש.
            self.model.zero_grad()

            # חישוב פלט המודל עבור האצווה הנוכחית.
            outputs = self.model(images)

            # חישוב cross entropy loss מול התוויות האמיתיות.
            loss = F.cross_entropy(outputs, labels)

            # חישוב הגרדיאנטים של ה-loss ביחס לפרמטרים.
            loss.backward()

            # מעבר על כל פרמטרי המודל כדי לצבור את ריבועי הגרדיאנטים.
            for n, p in self.model.named_parameters():

                # בדיקה שהפרמטר ניתן ללמידה ושאכן חושב עבורו גרדיאנט.
                if p.requires_grad and p.grad is not None:

                    # הוספת ריבוע הגרדיאנט ל-Fisher; ריבוע גדול מצביע על פרמטר חשוב יותר.
                    fisher[n] += p.grad.detach() ** 2

            # הגדלת מונה האצוות לאחר עיבוד batch אחד.
            batch_count += 1

        # מעבר על כל רכיבי Fisher לצורך נרמול.
        for n in fisher:

            # חלוקה במספר האצוות כדי לקבל ממוצע ולא סכום מצטבר.
            fisher[n] /= batch_count

        # החזרת Fisher האלכסוני המחושב.
        return fisher

    # אחרי כל משימה: מחשבים Fisher חדש ומעדכנים את נקודת העוגן של EWC
    # פעולה זו מתבצעת בסיום task, כדי לשמר את הידע שנלמד לפני מעבר למשימה הבאה.
    def update_after_task(self, dataloader):

        # חישוב Fisher עבור המשימה שזה עתה הסתיימה.
        current_fisher = self._diag_fisher(dataloader)

        # שמירת הפרמטרים הנוכחיים כנקודת העוגן החדשה.
        current_params = self._snapshot_params()

        # אם זו המשימה הראשונה, עדיין אין Fisher קודם.
        if self.fisher is None:

            # שמירת Fisher הראשון.
            self.fisher = current_fisher

            # שמירת הפרמטרים הראשונים כ-star parameters.
            self.star_params = current_params

        # אם כבר קיים Fisher ממשימות קודמות, מעדכנים אותו.
        else:

            # מעבר על כל הפרמטרים שכבר קיימים ב-Fisher.
            for n in self.fisher:

                # צבירת Fisher חדש על גבי Fisher קודם, כדי לחזק שימור של פרמטרים חשובים.
                self.fisher[n] = self.fisher[n] + current_fisher[n]

            # שומרים את אותו אופן פעולה כמו בקוד שנתן את התוצאה הכי טובה
            # עדכון נקודת העוגן לפרמטרים של המודל לאחר המשימה האחרונה.
            self.star_params = current_params

    # מחשב את איבר הענישה של EWC שמונע מהמשקולות לזוז יותר מדי מהערכים הקודמים
    # penalty מחזיר ערך loss נוסף שמתווסף ל-loss הרגיל בזמן אימון משימות חדשות.
    def penalty(self, model):

        # אם עדיין לא חושב Fisher או שלא נשמרו פרמטרים קודמים, אין מה להעניש.
        if self.fisher is None or self.star_params is None:

            # החזרת tensor אפס על אותו device כדי לשמור על תאימות חישובית.
            return torch.tensor(0.0, device=device)

        # אתחול ערך ה-penalty.
        loss = 0.0

        # מעבר על כל פרמטרי המודל הנוכחי.
        for n, p in model.named_parameters():

            # חישוב penalty רק עבור פרמטרים שניתנים ללמידה.
            if p.requires_grad:

                # הוספת קנס ריבועי: Fisher כפול ריבוע המרחק מהפרמטרים שנשמרו.
                loss += (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()

        # החזרת ערך ה-penalty הכולל.
        return loss

# =========================================================
# פונקציות עזר לאימון ולהערכה
# כאן מוגדרות פונקציות כלליות שמשמשות גם את baseline וגם את EWC.
# =========================================================

# מחשב accuracy של מודל על DataLoader נתון
# הפונקציה משמשת למדידת ביצועים על test set או validation set.
def evaluate(model, loader):

    # מעבר למצב הערכה כדי לבטל התנהגות אימון כמו Dropout.
    model.eval()

    # מונה מספר תחזיות נכונות.
    correct = 0

    # מונה מספר דוגמאות כולל.
    total = 0

    # ביטול חישוב גרדיאנטים בזמן הערכה כדי לחסוך זיכרון וחישוב.
    with torch.no_grad():

        # מעבר על כל האצוות ב-loader.
        for images, labels in loader:

            # העברת התמונות ל-device המתאים.
            images = images.to(device, non_blocking=True)

            # העברת התוויות ל-device המתאים.
            labels = labels.to(device, non_blocking=True)

            # חישוב פלט המודל.
            outputs = model(images)

            # בחירת המחלקה עם הערך הגבוה ביותר כתחזית המודל.
            preds = outputs.argmax(dim=1)

            # ספירת מספר התחזיות הנכונות באצווה.
            correct += (preds == labels).sum().item()

            # הוספת מספר הדוגמאות באצווה לסך הכול.
            total += labels.size(0)

    # החזרת accuracy כשבר בין 0 ל-1.
    return correct / total


# מחשב ממוצע accuracy על כל המשימות שנלמדו עד עכשיו
# זו המדידה המרכזית ב-Figure B: ביצוע ממוצע על tasks שכבר נלמדו.
def evaluate_average_seen_tasks(model, loaders, seen_task_count):

    # רשימה שתכיל את ה-accuracy של כל task שנלמד עד כה.
    accs = []

    # מעבר על כל המשימות שכבר נלמדו.
    for t in range(seen_task_count):

        # הערכת המודל על task אחד והוספת התוצאה לרשימה.
        accs.append(evaluate(model, loaders[t]))

    # חישוב ממוצע accuracy על כל המשימות שנראו עד עכשיו.
    return float(np.mean(accs))


# מאמן את המודל epoch אחד; אם יש EWC מוסיפים penalty ל-loss
# זוהי פונקציית האימון הבסיסית שמריצה מעבר אחד על כל הדאטה של task מסוים.
def train_one_epoch(model, train_loader, optimizer, ewc_obj=None, ewc_lambda=0):

    # מעבר למצב אימון.
    model.train()

    # משתנה לצבירת loss לאורך כל ה-epoch.
    running_loss = 0.0

    # מעבר על כל האצוות בסט האימון.
    for images, labels in train_loader:

        # העברת התמונות ל-device.
        images = images.to(device, non_blocking=True)

        # העברת התוויות ל-device.
        labels = labels.to(device, non_blocking=True)

        # איפוס הגרדיאנטים לפני חישוב loss חדש.
        optimizer.zero_grad()

        # חישוב פלט המודל עבור האצווה.
        outputs = model(images)

        # חישוב loss רגיל של סיווג רב-מחלקתי.
        loss = F.cross_entropy(outputs, labels)

        # אם קיים אובייקט EWC, מוסיפים את penalty ל-loss.
        if ewc_obj is not None:

            # שילוב loss רגיל עם קנס EWC מוכפל ב-lambda.
            loss = loss + ewc_lambda * ewc_obj.penalty(model)

        # חישוב גרדיאנטים עבור ה-loss הכולל.
        loss.backward()

        # הגבלת נורמת הגרדיאנטים כדי למנוע עדכונים חדים מדי ולא יציבים.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # עדכון פרמטרי המודל לפי האופטימייזר.
        optimizer.step()

        # צבירת ערך ה-loss לצורך חישוב ממוצע epoch.
        running_loss += loss.item()

    # החזרת loss ממוצע לאותו epoch.
    return running_loss / len(train_loader)


# מאמן משימה אחת למספר קבוע של epochs
# הפונקציה עוטפת את train_one_epoch ומדפיסה התקדמות לכל epoch.
def train_task_fixed_epochs(model, train_loader, optimizer, epochs, label="", ewc_obj=None, ewc_lambda=0):

    # לולאה על מספר ה-epochs שהוגדרו למשימה.
    for epoch in range(epochs):

        # אימון epoch אחד וקבלת loss ממוצע.
        loss = train_one_epoch(

            # המודל שאותו מאמנים.
            model,

            # DataLoader של סט האימון למשימה הנוכחית.
            train_loader,

            # האופטימייזר שמעדכן את הפרמטרים.
            optimizer,

            # אובייקט EWC אם משתמשים בו.
            ewc_obj=ewc_obj,

            # חוזק ענישת EWC.
            ewc_lambda=ewc_lambda
        )

        # הדפסת סטטוס האימון עבור ה-epoch הנוכחי.
        print(f"      {label} Epoch {epoch+1}/{epochs} - loss={loss:.4f}")

# =========================================================
# הרצת שלושת הניסויים: Single-task, SGD+Dropout, EWC
# בחלק הזה מוגדרות שלוש הרצות שונות לצורך השוואה בגרף.
# =========================================================

# מאמן מודל נפרד לכל משימה — זה הקו השחור המקווקו בגרף
# מאחר שכל task מקבל מודל עצמאי, זו הערכת upper reference ללא שכחה בין משימות.
def run_single_task_reference(train_loaders, test_loaders):

    # רשימה לשמירת accuracy של כל מודל עצמאי על המשימה שלו.
    task_accs = []

    # מעבר על כל המשימות.
    for t in range(MAX_TASKS):

        # הדפסת הודעה על תחילת אימון single-task למשימה הנוכחית.
        print(f"\n[Single-task] Training task {t+1}/{MAX_TASKS}")

        # יצירת מודל MLP חדש לחלוטין עבור task אחד בלבד.
        model = MLP().to(device)

        # יצירת אופטימייזר SGD עבור המודל העצמאי.
        optimizer = optim.SGD(model.parameters(), lr=LR_SINGLE, momentum=MOMENTUM)

        # אימון המודל על המשימה הנוכחית בלבד.
        train_task_fixed_epochs(

            # המודל העצמאי למשימה.
            model,

            # DataLoader של האימון למשימה הנוכחית.
            train_loaders[t],

            # האופטימייזר של אותו מודל.
            optimizer,

            # מספר ה-epochs לאימון.
            epochs=MAX_EPOCHS,

            # תווית להדפסות הלוג.
            label="single"
        )

        # הערכת המודל על test set של אותה משימה.
        acc = evaluate(model, test_loaders[t])

        # שמירת accuracy של task זה.
        task_accs.append(acc)

        # הדפסת accuracy של המודל העצמאי.
        print(f"   Task {t+1} single-task acc: {acc:.4f}")

    # רשימה שתכיל ממוצע מצטבר של ביצועי single-task.
    cumulative = []

    # חישוב ממוצע מצטבר עבור 1 עד MAX_TASKS משימות.
    for k in range(1, MAX_TASKS + 1):

        # ממוצע הביצועים של כל המשימות עד k.
        cumulative.append(float(np.mean(task_accs[:k])))

    # החזרת התוצאה כ-NumPy array לצורך ציור ועיבוד.
    return np.array(cumulative)


# מאמן מודל אחד ברצף על כל המשימות עם SGD+Dropout — הקו הכחול
# זהו baseline שבו אין מנגנון ייעודי לשמירת ידע, ולכן צפויה שכחה ממשימות קודמות.
def run_sgd_dropout_sequential(train_loaders, test_loaders):

    # יצירת מודל אחד עם Dropout שילמד את כל המשימות ברצף.
    model = DropoutMLP().to(device)

    # יצירת SGD עם learning rate שמתאים ל-baseline של Dropout.
    optimizer = optim.SGD(model.parameters(), lr=LR_DROPOUT, momentum=MOMENTUM)

    # רשימה לשמירת accuracy ממוצע אחרי כל task.
    history = []

    # מעבר סדרתי על כל המשימות.
    for t in range(MAX_TASKS):

        # הדפסת הודעה על תחילת אימון task במודל הרציף.
        print(f"\n[SGD+Dropout] Training task {t+1}/{MAX_TASKS}")

        # אימון המודל על task הנוכחי.
        train_task_fixed_epochs(

            # המודל הרציף עם Dropout.
            model=model,

            # DataLoader של task הנוכחי.
            train_loader=train_loaders[t],

            # האופטימייזר המשותף לאורך כל הרצף.
            optimizer=optimizer,

            # מספר epochs למשימה.
            epochs=MAX_EPOCHS,

            # תווית להדפסה.
            label="dropout"
        )

        # חישוב accuracy ממוצע על כל המשימות שנלמדו עד כה.
        avg_acc = evaluate_average_seen_tasks(model, test_loaders, t + 1)

        # שמירת accuracy ממוצע אחרי task זה.
        history.append(avg_acc)

        # הדפסת תוצאת ההערכה לאחר מספר המשימות שנלמדו.
        print(f"   Test avg accuracy after {t+1} tasks: {avg_acc:.4f}")

    # החזרת ההיסטוריה כ-array.
    return np.array(history)


# מאמן מודל אחד ברצף עם EWC — הקו האדום
# כאן EWC אמור לצמצם שכחה קטסטרופלית באמצעות penalty על שינוי פרמטרים חשובים.
def run_ewc_sequential(train_loaders, test_loaders):

    # יצירת מודל MLP אחד שילמד את כל המשימות ברצף.
    model = MLP().to(device)

    # יצירת SGD עבור אימון EWC.
    optimizer = optim.SGD(model.parameters(), lr=LR_EWC, momentum=MOMENTUM)

    # יצירת אובייקט EWC שמנהל Fisher ו-star parameters.
    ewc_obj = ConsolidatedEWC(model)

    # רשימה לשמירת accuracy ממוצע אחרי כל task.
    history = []

    # מעבר סדרתי על כל המשימות.
    for t in range(MAX_TASKS):

        # הדפסת הודעה על תחילת אימון task עם EWC ועל ערך lambda.
        print(f"\n[EWC λ={EWC_LAMBDA}] Training task {t+1}/{MAX_TASKS}")

        # אימון המודל על task הנוכחי.
        train_task_fixed_epochs(

            # המודל הרציף עם EWC.
            model=model,

            # DataLoader של task הנוכחי.
            train_loader=train_loaders[t],

            # האופטימייזר של המודל.
            optimizer=optimizer,

            # מספר epochs למשימה.
            epochs=MAX_EPOCHS,

            # תווית להדפסה.
            label="EWC",

            # במשימה הראשונה אין עדיין ידע קודם, ולכן EWC מופעל רק מ-task שני ואילך.
            ewc_obj=ewc_obj if t > 0 else None,

            # חוזק ה-penalty של EWC.
            ewc_lambda=EWC_LAMBDA
        )

        # לאחר סיום task, מחשבים Fisher ושומרים פרמטרים עבור הגנה במשימות הבאות.
        ewc_obj.update_after_task(train_loaders[t])

        # הערכת accuracy ממוצע על כל המשימות שנלמדו עד עכשיו.
        avg_acc = evaluate_average_seen_tasks(model, test_loaders, t + 1)

        # שמירת התוצאה בהיסטוריה.
        history.append(avg_acc)

        # הדפסת accuracy ממוצע לאחר task זה.
        print(f"   Test avg accuracy after {t+1} tasks: {avg_acc:.4f}")

    # החזרת היסטוריית התוצאות כ-array.
    return np.array(history)

# =========================================================
# שרטוט Figure B
# כאן מצויר הגרף הסופי שמשווה בין שלוש הגישות.
# =========================================================

# משרטט את הגרף הסופי שמשווה בין EWC, SGD+dropout ו-single task
# ציר X מציין כמה tasks נלמדו, וציר Y מציין accuracy ממוצע על המשימות שנראו.
def plot_figure_b(single_hist, dropout_hist, ewc_hist):

    # יצירת ערכי ציר X מ-2 עד מספר המשימות, כדי להתמקד אחרי שהצטברה למידה רציפה.
    x = list(range(2, MAX_TASKS + 1))

    # יצירת figure בגודל מתאים לגרף דמוי Figure B.
    plt.figure(figsize=(6.2, 4.2))

    # ציור עקומת EWC בצבע אדום.
    plt.plot(x, ewc_hist[1:], color="red", linewidth=2.4, label=f"EWC λ={EWC_LAMBDA}")

    # ציור עקומת SGD+dropout בצבע כחול.
    plt.plot(x, dropout_hist[1:], color="royalblue", linewidth=2.4, label="SGD+dropout")

    # ציור עקומת single-task reference כקו שחור מקווקו.
    plt.plot(

        # ערכי ציר X.
        x,

        # תוצאות single-task החל מהנקודה השנייה.
        single_hist[1:],

        # צבע הקו.
        color="black",

        # סגנון קו מקווקו.
        linestyle="--",

        # עובי הקו.
        linewidth=2.0,

        # שם העקומה במקרא.
        label="single task performance"
    )

    # הגדרת שם ציר X.
    plt.xlabel("Number of tasks")

    # הגדרת שם ציר Y.
    plt.ylabel("Fraction correct")

    # הגבלת ציר Y לטווח דיוקים גבוהים, בדומה להצגת הגרף במאמר.
    plt.ylim(0.80, 1.0)

    # הגבלת ציר X למספר המשימות הרלוונטי.
    plt.xlim(2, MAX_TASKS)

    # הוספת מקרא ללא מסגרת.
    plt.legend(frameon=False)

    # סידור אוטומטי של רכיבי הגרף כדי למנוע חפיפות.
    plt.tight_layout()

    # שמירת הגרף לקובץ PNG בשם הכולל את ערך lambda.
    plt.savefig(f"figure_B_lambda_{EWC_LAMBDA}.png", dpi=300)

    # הצגת הגרף על המסך.
    plt.show()

# =========================================================
# נקודת ההרצה הראשית של הקוד
# מכאן מתחילה הריצה בפועל כאשר מפעילים את הקובץ ישירות.
# =========================================================

# בדיקה שהקוד הראשי ירוץ רק כאשר הקובץ מופעל ישירות, ולא כאשר הוא מיובא כמודול.
if __name__ == "__main__":

    # שמירת זמן ההתחלה הכולל של הניסוי.
    total_start = time.time()

    # הדפסת כותרת לניסוי המבוקר של Figure B.
    print("\n========== Controlled Figure B test ==========")

    # הדפסת רוחב השכבות החבויות שנבחר.
    print(f"WIDTH={WIDTH}")

    # הדפסת learning rate של baseline עם Dropout.
    print(f"LR_DROPOUT={LR_DROPOUT}")

    # הדפסת learning rate של EWC.
    print(f"LR_EWC={LR_EWC}")

    # הדפסת חוזק ענישת EWC.
    print(f"EWC_LAMBDA={EWC_LAMBDA}")

    # רשימה לשמירת תוצאות single-task עבור כל seed.
    single_runs = []

    # רשימה לשמירת תוצאות SGD+Dropout עבור כל seed.
    dropout_runs = []

    # רשימה לשמירת תוצאות EWC עבור כל seed.
    ewc_runs = []

    # מעבר על כל ה-seeds שהוגדרו.
    for seed in SEEDS:

        # שמירת זמן התחלה עבור ה-seed הנוכחי.
        start = time.time()

        # הדפסת קו הפרדה לפני הרצת seed חדש.
        print("\n" + "=" * 80)

        # הדפסת מספר ה-seed הנוכחי.
        print(f"RUNNING SEED {seed}")

        # הדפסת קו הפרדה נוסף.
        print("=" * 80)

        # קביעת כל ה-seeds כדי לשמור על שחזוריות.
        set_all_seeds(seed)

        # יצירת הפרמוטציות לכל משימות Permuted MNIST עבור seed זה.
        perms = make_permutations(seed, MAX_TASKS)

        # בניית DataLoaders עבור train, validation ו-test לכל task.
        train_loaders, val_loaders, test_loaders = build_task_loaders(perms, seed)

        # הרצת ניסוי single-task ושמירת ההיסטוריה שלו.
        single_hist = run_single_task_reference(train_loaders, test_loaders)

        # הרצת ניסוי SGD+Dropout הרציף ושמירת ההיסטוריה שלו.
        dropout_hist = run_sgd_dropout_sequential(train_loaders, test_loaders)

        # הרצת ניסוי EWC הרציף ושמירת ההיסטוריה שלו.
        ewc_hist = run_ewc_sequential(train_loaders, test_loaders)

        # שמירת תוצאות single-task עבור seed זה.
        single_runs.append(single_hist)

        # שמירת תוצאות Dropout עבור seed זה.
        dropout_runs.append(dropout_hist)

        # שמירת תוצאות EWC עבור seed זה.
        ewc_runs.append(ewc_hist)

        # הדפסת זמן הריצה של ה-seed בדקות.
        print(f"\nSeed {seed} finished in {(time.time() - start) / 60:.2f} min")

    # חישוב ממוצע תוצאות single-task על פני כל ה-seeds.
    single_avg = np.mean(np.stack(single_runs, axis=0), axis=0)

    # חישוב ממוצע תוצאות Dropout על פני כל ה-seeds.
    dropout_avg = np.mean(np.stack(dropout_runs, axis=0), axis=0)

    # חישוב ממוצע תוצאות EWC על פני כל ה-seeds.
    ewc_avg = np.mean(np.stack(ewc_runs, axis=0), axis=0)

    # הדפסת כותרת לפני הצגת המערכים הסופיים.
    print("\n========== FINAL ARRAYS ==========")

    # הדפסת תוצאות single-task לאחר עיגול ל-4 ספרות.
    print("single:", [round(x, 4) for x in single_avg])

    # הדפסת תוצאות Dropout לאחר עיגול ל-4 ספרות.
    print("dropout:", [round(x, 4) for x in dropout_avg])

    # הדפסת תוצאות EWC לאחר עיגול ל-4 ספרות.
    print("EWC:", [round(x, 4) for x in ewc_avg])

    # ציור הגרף הסופי של Figure B.
    plot_figure_b(single_avg, dropout_avg, ewc_avg)

    # הדפסת זמן הריצה הכולל של כל הניסוי בדקות.
    print(f"\nTotal time: {(time.time() - total_start) / 60:.2f} min")
