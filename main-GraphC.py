# =========================================================
# Graph C reproduction - improved version
# גרסה משופרת לשחזור Figure C מהמאמר על EWC ו-Permuted MNIST.
# מתאים לשחזור Figure C מהמאמר:
# הקוד בודק את מידת החפיפה בין ערכי Fisher בשכבות שונות של הרשת.
# Fisher overlap לפי עומק שכבה
# כלומר, בודקים עד כמה שתי משימות שונות משתמשות בפרמטרים דומים בכל שכבה.
# low = שתי פרמוטציות שונות של ריבוע 8x8 במרכז
# מצב low מייצג שינוי קטן יחסית בתמונה, כי רק אזור קטן במרכז עובר permutation.
# high = שתי פרמוטציות שונות של ריבוע 26x26 במרכז
# מצב high מייצג שינוי גדול יותר, כי כמעט כל התמונה עוברת permutation.
# =========================================================

# ייבוא מודול time למדידת זמני הריצה של כל seed ושל כל הניסוי.
import time

# ייבוא copy לצורך שמירת עותקים של מצב המודל, למשל best_state או init_state.
import copy

# ייבוא NumPy לעבודה עם מערכים, חישוב ממוצעים וארגון תוצאות.
import numpy as np

# ייבוא PyTorch לעבודה עם tensors, מודלים, גרדיאנטים והרצה על CPU/GPU.
import torch

# ייבוא מודול הרשתות הנוירוניות של PyTorch.
import torch.nn as nn

# ייבוא פונקציות שימושיות כמו ReLU ו-cross_entropy.
import torch.nn.functional as F

# ייבוא אופטימייזרים, ובפרט SGD עם momentum.
import torch.optim as optim

# ייבוא כלים לטעינת נתונים, יצירת datasets, חלוקה ל-train/validation, ויצירת subset לחישוב Fisher.
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

# ייבוא torchvision כדי לטעון את דאטהסט MNIST.
import torchvision

# ייבוא matplotlib לציור הגרף הסופי של Figure C.
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# חלק זה מרכז את הגדרות הניסוי וה-hyperparameters.
# לפי המאמר Figure C:
# ההגדרות כאן מנסות להתאים למבנה הניסוי של Figure C במאמר.
# 6 שכבות חבויות, רוחב 100, 100 epochs, ללא dropout וללא early stopping
# כלומר המודל עמוק יחסית, אך כל שכבה חבויה צרה יחסית.
# =========================================================

# רשימת seeds להרצות חוזרות, כדי למצע תוצאות ולהקטין תלות באקראיות.
SEEDS = [0, 1, 2, 3, 4]

# גודל batch רגיל בזמן אימון המודל.
BATCH_SIZE = 256

# גודל batch לחישוב Fisher; ערך 1 מאפשר לחשב גרדיאנט לכל דוגמה בנפרד.
FISHER_BATCH_SIZE = 1       # חשוב: Fisher יותר מדויק כשמחשבים פר דוגמה

# מספר הנוירונים בכל שכבה חבויה.
WIDTH = 100

# מספר השכבות החבויות במודל.
NUM_HIDDEN_LAYERS = 6

# מספר epochs לאימון כל מודל על כל task.
EPOCHS = 100

# קצב הלמידה (learning rate) עבור SGD.
LR = 0.001

# momentum עבור SGD, שמסייע לייצב ולהאיץ את תהליך האימון.
MOMENTUM = 0.9

# אחוז הדאטה שיופרד ל-validation מתוך סט האימון.
VAL_FRACTION = 0.1

# כדי לא להפוך את Fisher לאיטי מדי.
# חישוב Fisher מלא על כל הדאטה יכול להיות יקר מאוד, לכן משתמשים ב-subset.
# אפשר להעלות ל-10000 אם יש זמן.
# הגדלת הערך תשפר את הדיוק הסטטיסטי אך תאריך את זמן הריצה.
FISHER_SUBSET_SIZE = 20000

# מספר worker-ים לטעינת הנתונים; 0 מתאים לסביבות פשוטות כמו Windows או notebooks.
NUM_WORKERS = 0

# בחירת התקן החישוב: GPU אם CUDA זמינה, אחרת CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# הדפסת ההתקן שבו הקוד ירוץ.
print("Using device:", device)

# בדיקה האם יש GPU זמין.
if torch.cuda.is_available():

    # הדפסת שם ה-GPU, אם קיים.
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# SEED
# חלק זה אחראי על שחזוריות (reproducibility) של ההרצה.
# =========================================================

# פונקציה שמקבעת את כל מקורות האקראיות המרכזיים לפי seed נתון.
def set_all_seeds(seed):

    # קביעת seed עבור NumPy.
    np.random.seed(seed)

    # קביעת seed עבור PyTorch על CPU.
    torch.manual_seed(seed)

    # אם CUDA זמינה, מקבעים גם את האקראיות של GPU.
    if torch.cuda.is_available():

        # קביעת seed לכל התקני CUDA הזמינים.
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA
# חלק זה טוען את MNIST ומכין אותו לעבודה עם Permuted MNIST.
# =========================================================

# פונקציה שטוענת את MNIST המקורי ומחזירה train/test כ-tensors מנורמלים.
def load_base_mnist():

    # טעינת סט האימון המקורי של MNIST.
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)

    # טעינת סט הבדיקה המקורי של MNIST.
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    # המרת תמונות האימון ל-float, נרמול לטווח 0 עד 1, והוספת ממד channel.
    x_train = train_ds.data.float().div(255.0).unsqueeze(1)

    # המרת תמונות הבדיקה באותה צורה.
    x_test = test_ds.data.float().div(255.0).unsqueeze(1)

    # נרמול סטנדרטי של MNIST
    # הפחתת הממוצע וחלוקה בסטיית התקן המקובלים עבור MNIST.
    x_train = (x_train - 0.1307) / 0.3081

    # נרמול סט הבדיקה באותו אופן כדי לשמור על עקביות עם סט האימון.
    x_test = (x_test - 0.1307) / 0.3081

    # שליפת תוויות האימון והמרתן ל-long, כפי שנדרש עבור cross_entropy.
    y_train = train_ds.targets.long()

    # שליפת תוויות הבדיקה והמרתן ל-long.
    y_test = test_ds.targets.long()

    # החזרת התמונות והתוויות של train ושל test.
    return x_train, y_train, x_test, y_test


# טעינת MNIST פעם אחת ושמירתו כבסיס לכל הפרמוטציות.
BASE_X_TRAIN, BASE_Y_TRAIN, BASE_X_TEST, BASE_Y_TEST = load_base_mnist()


# =========================================================
# PERMUTATIONS
# חלק זה יוצר permutation חלקית של אזורים בתמונה.
# =========================================================

# פונקציה שמחזירה את האינדקסים של ריבוע מרכזי בתוך תמונת 28x28.
def make_square_indices(square_size):
    """
    מחזיר את האינדקסים של ריבוע במרכז תמונת 28x28.
    למשל:
    square_size=8  -> ריבוע קטן במרכז
    square_size=26 -> כמעט כל התמונה
    """

    # חישוב נקודת ההתחלה של הריבוע כך שיהיה ממורכז בתמונה.
    start = (28 - square_size) // 2

    # חישוב נקודת הסיום של הריבוע.
    end = start + square_size

    # רשימה שתכיל את האינדקסים הלינאריים של הפיקסלים בריבוע.
    idxs = []

    # מעבר על שורות הריבוע המרכזי.
    for r in range(start, end):

        # מעבר על עמודות הריבוע המרכזי.
        for c in range(start, end):

            # המרת מיקום דו-ממדי לאינדקס חד-ממדי בתוך וקטור באורך 784.
            idxs.append(r * 28 + c)

    # החזרת האינדקסים כ-tensor מסוג long.
    return torch.tensor(idxs, dtype=torch.long)


# פונקציה שיוצרת permutation שבה רק ריבוע מרכזי מסוים עובר ערבוב.
def make_partial_square_permutation(square_size, seed):
    """
    יוצר פרמוטציה שבה רק הריבוע המרכזי עובר ערבוב.
    שאר הפיקסלים נשארים במקום.
    """

    # יצירת generator מקומי כדי לשלוט באקראיות של הפרמוטציה.
    g = torch.Generator()

    # קביעת seed עבור ה-generator.
    g.manual_seed(seed)

    # יצירת permutation התחלתית זהותית, שבה כל פיקסל נשאר במקומו.
    perm = torch.arange(784)

    # קבלת האינדקסים של הריבוע המרכזי שאותו רוצים לערבב.
    square_idxs = make_square_indices(square_size)

    # יצירת סדר אקראי חדש רק עבור הפיקסלים שנמצאים בתוך הריבוע.
    shuffled_square = square_idxs[torch.randperm(len(square_idxs), generator=g)]

    # החלפת מיקומי הפיקסלים בתוך הריבוע בלבד, בלי לשנות את שאר הפיקסלים.
    perm[square_idxs] = shuffled_square

    # החזרת permutation מלאה באורך 784.
    return perm


# פונקציה שמחילה permutation על סט תמונות.
def apply_permutation(x, perm):
    """
    מפעיל את הפרמוטציה על התמונות.
    """

    # הפיכת כל תמונה לווקטור שטוח באורך 784.
    flat = x.view(x.size(0), -1)

    # סידור מחדש של הפיקסלים לפי ה-permutation.
    flat = flat[:, perm]

    # החזרת התמונות לצורה המקורית: מספר דוגמאות, channel אחד, 28x28.
    return flat.view(-1, 1, 28, 28)


# פונקציה שבונה loaders עבור task יחיד שמוגדר על ידי permutation מסוימת.
def build_loaders_for_perm(perm, seed):
    """
    בונה loaders עבור משימה אחת:
    train / validation / test / fisher
    """

    # יצירת סט אימון לאחר החלת ה-permutation על התמונות.
    x_train = apply_permutation(BASE_X_TRAIN, perm)

    # יצירת סט בדיקה לאחר החלת אותה permutation.
    x_test = apply_permutation(BASE_X_TEST, perm)

    # יצירת TensorDataset עבור האימון עם התוויות המקוריות.
    full_train_ds = TensorDataset(x_train, BASE_Y_TRAIN)

    # יצירת TensorDataset עבור הבדיקה.
    test_ds = TensorDataset(x_test, BASE_Y_TEST)

    # חישוב גודל validation set לפי VAL_FRACTION.
    val_size = int(len(full_train_ds) * VAL_FRACTION)

    # חישוב גודל train set לאחר הפרדת validation.
    train_size = len(full_train_ds) - val_size

    # יצירת generator עבור חלוקה שחזורית בין train ל-validation.
    split_gen = torch.Generator()

    # קביעת seed לפיצול הדאטה.
    split_gen.manual_seed(seed + 999)

    # חלוקת סט האימון ל-train ול-validation.
    train_ds, val_ds = random_split(

        # הדאטהסט המלא שממנו מפצלים.
        full_train_ds,

        # גדלי הפיצול: אימון ואז validation.
        [train_size, val_size],

        # generator ששולט באקראיות של הפיצול.
        generator=split_gen
    )

    # יצירת DataLoader עבור האימון.
    train_loader = DataLoader(

        # סט האימון.
        train_ds,

        # גודל batch רגיל לאימון.
        batch_size=BATCH_SIZE,

        # ערבוב הדוגמאות בזמן אימון.
        shuffle=True,

        # מספר worker-ים לטעינת הנתונים.
        num_workers=NUM_WORKERS
    )

    # יצירת DataLoader עבור validation.
    val_loader = DataLoader(

        # סט ה-validation.
        val_ds,

        # גודל batch עבור validation.
        batch_size=BATCH_SIZE,

        # אין צורך לערבב בזמן validation.
        shuffle=False,

        # מספר worker-ים לטעינת הנתונים.
        num_workers=NUM_WORKERS
    )

    # יצירת DataLoader עבור test.
    test_loader = DataLoader(

        # סט הבדיקה.
        test_ds,

        # גודל batch עבור בדיקה.
        batch_size=BATCH_SIZE,

        # אין צורך בערבוב בזמן בדיקה.
        shuffle=False,

        # מספר worker-ים לטעינת הנתונים.
        num_workers=NUM_WORKERS
    )

    # Fisher מחושב על subset כדי לחסוך זמן
    # בחירת מספר הדוגמאות שישמשו לחישוב Fisher, עד לגודל המקסימלי שהוגדר.
    fisher_size = min(FISHER_SUBSET_SIZE, len(train_ds))

    # יצירת subset מתוך train set עבור חישוב Fisher.
    fisher_subset = Subset(train_ds, list(range(fisher_size)))

    # batch_size=1 כדי לקבל קירוב טוב יותר ל-average per-sample squared gradient
    # חישוב Fisher מדויק יותר כאשר מחשבים גרדיאנט לכל דוגמה בנפרד.
    fisher_loader = DataLoader(

        # ה-subset שעליו יחושב Fisher.
        fisher_subset,

        # batch size קטן במיוחד לחישוב Fisher פר דוגמה.
        batch_size=FISHER_BATCH_SIZE,

        # אין צורך לערבב, כי רק מחשבים סטטיסטיקה.
        shuffle=False,

        # מספר worker-ים לטעינת הדאטה.
        num_workers=NUM_WORKERS
    )

    # החזרת כל ה-loaders הדרושים למשימה.
    return train_loader, val_loader, test_loader, fisher_loader


# =========================================================
# MODEL
# הגדרת מודל MLP עם 6 שכבות חבויות.
# =========================================================

# מחלקה שמגדירה רשת fully connected עם מספר שכבות חבויות.
class SixLayerMLP(nn.Module):
    """
    רשת fully-connected עם 6 שכבות חבויות.
    זה מתאים לטבלת ההיפרפרמטרים של Figure C במאמר.
    """

    # אתחול המודל עם רוחב שכבה ומספר שכבות חבויות.
    def __init__(self, width=WIDTH, num_hidden=NUM_HIDDEN_LAYERS):

        # אתחול מחלקת הבסיס nn.Module.
        super().__init__()

        # שכבה שמיישרת תמונת 28x28 לווקטור באורך 784.
        self.flatten = nn.Flatten()

        # ModuleList מאפשרת לשמור רשימה של שכבות כך ש-PyTorch יזהה את הפרמטרים שלהן.
        self.hidden_layers = nn.ModuleList()

        # ממד הקלט הראשוני הוא 784 פיקסלים.
        in_dim = 784

        # יצירת מספר שכבות חבויות לפי num_hidden.
        for _ in range(num_hidden):

            # הוספת שכבה לינארית חדשה מ-in_dim ל-width.
            self.hidden_layers.append(nn.Linear(in_dim, width))

            # אחרי השכבה הראשונה, ממד הקלט לשכבה הבאה הוא width.
            in_dim = width

        # שכבת פלט שממפה מ-width ל-10 מחלקות של MNIST.
        self.output = nn.Linear(width, 10)

    # הגדרת מעבר קדימה של המודל.
    def forward(self, x):

        # הפיכת התמונה לווקטור שטוח.
        x = self.flatten(x)

        # העברת הקלט דרך כל השכבות החבויות.
        for layer in self.hidden_layers:

            # החלת שכבה לינארית ואחריה ReLU.
            x = F.relu(layer(x))

        # החזרת פלט המודל, כלומר logits עבור 10 ספרות.
        return self.output(x)


# =========================================================
# TRAIN / EVAL
# פונקציות לאימון ולהערכת המודל.
# =========================================================

# פונקציה שמחשבת accuracy של מודל על loader נתון.
def evaluate(model, loader):

    # מעבר למצב הערכה.
    model.eval()

    # מונה תחזיות נכונות.
    correct = 0

    # מונה את מספר הדוגמאות הכולל.
    total = 0

    # ביטול חישוב גרדיאנטים בזמן הערכה כדי לחסוך זיכרון וחישוב.
    with torch.no_grad():

        # מעבר על כל ה-batches ב-loader.
        for images, labels in loader:

            # העברת התמונות ל-device המתאים.
            images = images.to(device)

            # העברת התוויות ל-device המתאים.
            labels = labels.to(device)

            # חישוב פלט המודל.
            outputs = model(images)

            # בחירת המחלקה עם הערך הגבוה ביותר כתחזית.
            preds = outputs.argmax(dim=1)

            # ספירת התחזיות הנכונות באצווה.
            correct += (preds == labels).sum().item()

            # עדכון מספר הדוגמאות הכולל.
            total += labels.size(0)

    # החזרת accuracy כשבר בין 0 ל-1.
    return correct / total


# פונקציה שמאמנת מודל על task אחד.
def train_model(model, train_loader, val_loader, label):
    """
    אימון רגיל ל-100 epochs.
    אין early stopping, אבל כן נשמור את המודל עם ה-validation accuracy הכי טוב,
    כדי להימנע ממודל סופי גרוע במקרה של רעש באימון.
    """

    # יצירת אופטימייזר SGD עם learning rate ו-momentum שהוגדרו.
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # אתחול accuracy הטוב ביותר ב-validation לערך נמוך.
    best_val = -1.0

    # שמירת מצב התחלתי של המודל כ-best_state.
    best_state = copy.deepcopy(model.state_dict())

    # לולאת אימון לאורך מספר epochs קבוע.
    for epoch in range(EPOCHS):

        # מעבר למצב אימון.
        model.train()

        # משתנה לצבירת loss לאורך epoch.
        running_loss = 0.0

        # מעבר על כל ה-batches בסט האימון.
        for images, labels in train_loader:

            # העברת התמונות ל-device.
            images = images.to(device)

            # העברת התוויות ל-device.
            labels = labels.to(device)

            # איפוס גרדיאנטים לפני חישוב חדש.
            optimizer.zero_grad()

            # חישוב פלט המודל.
            outputs = model(images)

            # חישוב cross entropy loss מול התוויות האמיתיות.
            loss = F.cross_entropy(outputs, labels)

            # חישוב גרדיאנטים.
            loss.backward()

            # הגבלת נורמת הגרדיאנטים כדי למנוע עדכונים גדולים מדי.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # עדכון פרמטרי המודל.
            optimizer.step()

            # צבירת loss לצורך הדפסה.
            running_loss += loss.item()

        # הערכת המודל על validation set בסוף epoch.
        val_acc = evaluate(model, val_loader)

        # אם accuracy ב-validation השתפר, שומרים את מצב המודל.
        if val_acc > best_val:

            # עדכון accuracy הטוב ביותר.
            best_val = val_acc

            # שמירת עותק של הפרמטרים במצב הטוב ביותר.
            best_state = copy.deepcopy(model.state_dict())

        # הדפסת סטטוס האימון עבור epoch נוכחי.
        print(
            f"[{label}] Epoch {epoch+1}/{EPOCHS} "
            f"- loss={running_loss / len(train_loader):.4f} "
            f"- val_acc={val_acc:.4f}"
        )

    # טעינת המודל עם ביצועי ה-validation הטובים ביותר.
    model.load_state_dict(best_state)

    # החזרת המודל המאומן.
    return model


# =========================================================
# FISHER
# חישוב Fisher Information וחפיפה בין Fisher של שתי משימות.
# =========================================================

# פונקציה שמחשבת Fisher אלכסוני עבור מודל ודאטה נתונים.
def compute_diag_fisher(model, loader):
    """
    מחשב diagonal Fisher:
    F_i ≈ average over samples of (dL/dtheta_i)^2

    חשוב:
    כאן loader עם batch_size=1, כדי שלא נרבע גרדיאנט ממוצע של batch שלם.
    """

    # יצירת מילון Fisher עם tensor אפסים עבור כל פרמטר ניתן-למידה.
    fisher = {

        # יצירת tensor אפסים באותה צורה של הפרמטר ועל אותו device.
        n: torch.zeros_like(p, device=device)

        # מעבר על כל הפרמטרים של המודל.
        for n, p in model.named_parameters()

        # שמירה רק של פרמטרים שנלמדים באימון.
        if p.requires_grad
    }

    # מעבר למצב הערכה.
    model.eval()

    # מונה את מספר הדוגמאות שעליהן חושב Fisher.
    sample_count = 0

    # מעבר על הדוגמאות ב-loader, בדרך כלל batch_size=1.
    for images, labels in loader:

        # העברת התמונות ל-device.
        images = images.to(device)

        # העברת התוויות ל-device.
        labels = labels.to(device)

        # איפוס גרדיאנטים בצורה יעילה.
        model.zero_grad(set_to_none=True)

        # חישוב פלט המודל.
        outputs = model(images)

        # חישוב loss עבור הדוגמה או האצווה הנוכחית.
        loss = F.cross_entropy(outputs, labels)

        # חישוב גרדיאנטים ביחס לפרמטרים.
        loss.backward()

        # מעבר על כל הפרמטרים לצורך צבירת ריבועי הגרדיאנטים.
        for n, p in model.named_parameters():

            # בדיקה שהפרמטר ניתן ללמידה ושקיים עבורו גרדיאנט.
            if p.requires_grad and p.grad is not None:

                # הוספת ריבוע הגרדיאנט ל-Fisher.
                fisher[n] += p.grad.detach() ** 2

        # הגדלת מונה הדוגמאות.
        sample_count += 1

    # מעבר על כל רכיבי Fisher לצורך נרמול.
    for n in fisher:

        # חלוקה במספר הדוגמאות; max מונע חלוקה באפס במקרה חריג.
        fisher[n] /= max(sample_count, 1)

    # החזרת Fisher האלכסוני.
    return fisher


# פונקציה שמחלצת את וקטורי Fisher של משקלי השכבות החבויות בלבד.
def fisher_hidden_layer_weight_vectors(fisher_dict):
    """
    מחלץ רק את ה-weight של כל שכבה חבויה.
    לא משתמשים ב-bias כי הוא יכול לטשטש את ההבדל בין השכבות.
    """

    # רשימה שתכיל וקטור Fisher שטוח עבור כל שכבה חבויה.
    layer_vectors = []

    # מעבר על כל השכבות החבויות.
    for i in range(NUM_HIDDEN_LAYERS):

        # בניית שם הפרמטר כפי שהוא מופיע ב-state_dict וב-named_parameters.
        name = f"hidden_layers.{i}.weight"

        # בדיקה שה-Fisher אכן מכיל ערכים עבור שכבה זו.
        if name not in fisher_dict:

            # אם חסר Fisher לשכבה, זורקים שגיאה ברורה.
            raise RuntimeError(f"Missing Fisher values for {name}")

        # שיטוח מטריצת המשקלים של השכבה לווקטור והוספתה לרשימה.
        layer_vectors.append(fisher_dict[name].reshape(-1))

    # החזרת רשימת וקטורי Fisher לפי שכבה.
    return layer_vectors


# פונקציה שמחשבת overlap בין שני וקטורי Fisher אלכסוניים.
def fisher_overlap_diag(f1, f2, eps=1e-12):
    """
    חישוב overlap עבור Fisher דיאגונלי.

    במאמר מגדירים:
    overlap = 1 - d^2
    עבור מטריצות דיאגונליות, זה שקול לסכום:
    sum sqrt(f1_norm * f2_norm)

    כלומר זה כמו Bhattacharyya coefficient בין שני וקטורי Fisher מנורמלים.
    """

    # מבטיחים שערכי Fisher לא יהיו שליליים מבחינה מספרית.
    f1 = f1.clamp_min(0)

    # מבטיחים שגם ה-Fisher השני אינו מכיל ערכים שליליים.
    f2 = f2.clamp_min(0)

    # סכום ערכי Fisher הראשון.
    s1 = f1.sum()

    # סכום ערכי Fisher השני.
    s2 = f2.sum()

    # אם אחד הסכומים כמעט אפס, אין בסיס יציב לחישוב overlap.
    if s1.item() < eps or s2.item() < eps:

        # במקרה כזה מחזירים overlap אפסי.
        return 0.0

    # נרמול Fisher הראשון כך שסכומו יהיה 1.
    f1n = f1 / s1

    # נרמול Fisher השני כך שסכומו יהיה 1.
    f2n = f2 / s2

    # חישוב Bhattacharyya-like overlap בין שני וקטורי Fisher מנורמלים.
    overlap = torch.sqrt(f1n * f2n).sum()

    # החזרת ערך overlap כ-float רגיל של Python.
    return float(overlap.item())


# פונקציה שמחשבת overlap לכל שכבה חבויה בין שני Fisher dictionaries.
def compute_layerwise_overlap(fisher_a, fisher_b):
    """
    מחשב overlap לכל אחת מ-6 השכבות.
    """

    # חילוץ וקטורי Fisher עבור השכבות החבויות במשימה הראשונה.
    layers_a = fisher_hidden_layer_weight_vectors(fisher_a)

    # חילוץ וקטורי Fisher עבור השכבות החבויות במשימה השנייה.
    layers_b = fisher_hidden_layer_weight_vectors(fisher_b)

    # רשימה לשמירת overlap עבור כל שכבה.
    overlaps = []

    # מעבר זוגי על השכבות של שתי המשימות.
    for fa, fb in zip(layers_a, layers_b):

        # חישוב overlap עבור שכבה אחת והוספתו לרשימה.
        overlaps.append(fisher_overlap_diag(fa, fb))

    # החזרת רשימת overlap לפי עומק שכבה.
    return overlaps


# =========================================================
# RUN ONE SEED
# הרצת כל הניסוי עבור seed אחד.
# =========================================================

# פונקציה שמאמנת מודל על task אחד ומחזירה accuracy ו-Fisher.
def train_task_and_compute_fisher(perm, seed, init_state, label):
    """
    מאמן מודל על משימה אחת ומחזיר Fisher.
    כל המשימות מתחילות מאותו initialization כדי שההשוואה תהיה הוגנת.
    """

    # בניית loaders עבור task שמוגדר על ידי permutation נתונה.
    train_loader, val_loader, test_loader, fisher_loader = build_loaders_for_perm(perm, seed)

    # יצירת מודל חדש.
    model = SixLayerMLP().to(device)

    # טעינת אותו initialization כדי שכל ההשוואות יתחילו מאותה נקודת התחלה.
    model.load_state_dict(init_state)

    # אימון המודל על task הנוכחי.
    model = train_model(model, train_loader, val_loader, label=label)

    # חישוב accuracy על test set של אותו task.
    acc = evaluate(model, test_loader)

    # חישוב Fisher עבור המודל המאומן.
    fisher = compute_diag_fisher(model, fisher_loader)

    # החזרת הדיוק ו-Fisher.
    return acc, fisher


# פונקציה שמריצה את כל ניסוי Figure C עבור seed אחד.
def run_one_seed(seed):

    # הדפסת קו הפרדה ותחילת הרצת seed.
    print("\n" + "=" * 80)

    # הדפסת מספר ה-seed הנוכחי.
    print(f"RUNNING SEED {seed}")

    # הדפסת קו הפרדה נוסף.
    print("=" * 80)

    # קביעת seeds כדי לשלוט באקראיות.
    set_all_seeds(seed)

    # שתי פרמוטציות שונות עבור low ושתי פרמוטציות שונות עבור high
    # כך משווים בין זוג משימות בעלות אותו אחוז שינוי, אך עם ערבוב שונה.
    # זה יותר קרוב לרעיון של השוואה בין שתי משימות שונות באותה דרגת שינוי.
    # מצב low: שתי פרמוטציות שונות של ריבוע קטן בגודל 8x8.
    low_perm_1 = make_partial_square_permutation(square_size=8, seed=seed + 100)

    # פרמוטציה שנייה ועצמאית עבור low.
    low_perm_2 = make_partial_square_permutation(square_size=8, seed=seed + 101)

    # מצב high: פרמוטציה ראשונה של ריבוע גדול בגודל 26x26.
    high_perm_1 = make_partial_square_permutation(square_size=26, seed=seed + 200)

    # פרמוטציה שנייה ועצמאית עבור high.
    high_perm_2 = make_partial_square_permutation(square_size=26, seed=seed + 201)

    # אותה אתחול לכל המודלים באותו seed
    # מקבעים שוב seed לפני יצירת מודל ההתחלה כדי להבטיח initialization זהה.
    set_all_seeds(seed)

    # יצירת מודל התחלתי שממנו נעתיק את אותם משקלים לכל המשימות.
    init_model = SixLayerMLP().to(device)

    # שמירת מצב ההתחלה של המודל.
    init_state = copy.deepcopy(init_model.state_dict())

    # Low pair
    # אימון מודל ראשון על משימת low הראשונה וחישוב Fisher.
    acc_low_1, fisher_low_1 = train_task_and_compute_fisher(

        # permutation של low A.
        low_perm_1,

        # seed עבור חלוקת הנתונים ואימון המשימה.
        seed + 1,

        # מצב התחלה זהה לכל המודלים באותו seed.
        init_state,

        # label להדפסות האימון.
        label=f"seed{seed}_low_8x8_A"
    )

    # אימון מודל שני על משימת low השנייה וחישוב Fisher.
    acc_low_2, fisher_low_2 = train_task_and_compute_fisher(

        # permutation של low B.
        low_perm_2,

        # seed שונה כדי להפריד את הריצה.
        seed + 2,

        # אותו מצב התחלה לצורך השוואה הוגנת.
        init_state,

        # label להדפסות.
        label=f"seed{seed}_low_8x8_B"
    )

    # High pair
    # אימון מודל ראשון על משימת high הראשונה וחישוב Fisher.
    acc_high_1, fisher_high_1 = train_task_and_compute_fisher(

        # permutation של high A.
        high_perm_1,

        # seed עבור הריצה.
        seed + 3,

        # אותו initialization.
        init_state,

        # label להדפסות.
        label=f"seed{seed}_high_26x26_A"
    )

    # אימון מודל שני על משימת high השנייה וחישוב Fisher.
    acc_high_2, fisher_high_2 = train_task_and_compute_fisher(

        # permutation של high B.
        high_perm_2,

        # seed עבור הריצה.
        seed + 4,

        # אותו initialization.
        init_state,

        # label להדפסות.
        label=f"seed{seed}_high_26x26_B"
    )

    # חישוב overlap בין שתי משימות low, לכל שכבה.
    overlap_low = compute_layerwise_overlap(fisher_low_1, fisher_low_2)

    # חישוב overlap בין שתי משימות high, לכל שכבה.
    overlap_high = compute_layerwise_overlap(fisher_high_1, fisher_high_2)

    # הדפסת כותרת לתוצאות ה-seed.
    print("\nSeed result:")

    # הדפסת accuracy של משימת low הראשונה.
    print(f"Low 8x8 A accuracy:   {acc_low_1:.4f}")

    # הדפסת accuracy של משימת low השנייה.
    print(f"Low 8x8 B accuracy:   {acc_low_2:.4f}")

    # הדפסת accuracy של משימת high הראשונה.
    print(f"High 26x26 A acc:     {acc_high_1:.4f}")

    # הדפסת accuracy של משימת high השנייה.
    print(f"High 26x26 B acc:     {acc_high_2:.4f}")

    # הדפסת ערכי overlap עבור מצב low.
    print(f"Low overlap:          {overlap_low}")

    # הדפסת ערכי overlap עבור מצב high.
    print(f"High overlap:         {overlap_high}")

    # החזרת overlap כ-NumPy arrays כדי לאפשר חישוב ממוצע על פני seeds.
    return np.array(overlap_low), np.array(overlap_high)


# =========================================================
# PLOT
# ציור הגרף הסופי של Figure C.
# =========================================================

# פונקציה שמציירת את Fisher overlap לפי עומק שכבה.
def plot_graph_c(overlap_low_avg, overlap_high_avg):

    # ציר X מייצג את עומק השכבה: שכבות 1 עד 6.
    x = list(range(1, NUM_HIDDEN_LAYERS + 1))

    # יצירת figure בגודל מתאים.
    plt.figure(figsize=(6.0, 4.2))

    # ציור עקומת low permutation.
    plt.plot(

        # ערכי עומק השכבה.
        x,

        # ממוצע overlap עבור low.
        overlap_low_avg,

        # צבע הקו.
        color="gray",

        # סגנון קו מקווקו.
        linestyle="--",

        # עובי הקו.
        linewidth=2.2,

        # סימון נקודות המדידה.
        marker="o",

        # שם העקומה במקרא.
        label="low % permutation"
    )

    # ציור עקומת high permutation.
    plt.plot(

        # ערכי עומק השכבה.
        x,

        # ממוצע overlap עבור high.
        overlap_high_avg,

        # צבע הקו.
        color="black",

        # סגנון קו מקווקו.
        linestyle="--",

        # עובי הקו.
        linewidth=2.2,

        # סימון נקודות המדידה.
        marker="o",

        # שם העקומה במקרא.
        label="high % permutation"
    )

    # הגדרת שם ציר X.
    plt.xlabel("Layer depth")

    # הגדרת שם ציר Y.
    plt.ylabel("Overlap in Fisher")

    # הצגת סימון לכל שכבה בציר X.
    plt.xticks(x)

    # הגבלת ציר Y לטווח האפשרי של overlap.
    plt.ylim(0.0, 1.0)

    # הגבלת ציר X לפי מספר השכבות.
    plt.xlim(1, NUM_HIDDEN_LAYERS)

    # הוספת מקרא ללא מסגרת.
    plt.legend(frameon=False)

    # סידור אוטומטי של הגרף כדי למנוע חפיפות.
    plt.tight_layout()

    # שם הקובץ שאליו יישמר הגרף.
    filename = "figure_C_improved.png"

    # שמירת הגרף לקובץ PNG באיכות גבוהה.
    plt.savefig(filename, dpi=300)

    # הצגת הגרף על המסך.
    plt.show()


# =========================================================
# MAIN
# נקודת הכניסה הראשית של הסקריפט.
# =========================================================

# הרצת הקוד הראשי רק כאשר הקובץ מופעל ישירות.
if __name__ == "__main__":

    # שמירת זמן ההתחלה של כל הניסוי.
    total_start = time.time()

    # הדפסת כותרת לתחילת הרצת Figure C.
    print("\n========== Running improved Figure C reproduction ==========")

    # הדפסת רשימת ה-seeds.
    print(f"SEEDS={SEEDS}")

    # הדפסת רוחב השכבות.
    print(f"WIDTH={WIDTH}")

    # הדפסת מספר השכבות החבויות.
    print(f"NUM_HIDDEN_LAYERS={NUM_HIDDEN_LAYERS}")

    # הדפסת מספר epochs.
    print(f"EPOCHS={EPOCHS}")

    # הדפסת learning rate.
    print(f"LR={LR}")

    # הדפסת גודל ה-subset לחישוב Fisher.
    print(f"FISHER_SUBSET_SIZE={FISHER_SUBSET_SIZE}")

    # הדפסת batch size לחישוב Fisher.
    print(f"FISHER_BATCH_SIZE={FISHER_BATCH_SIZE}")

    # רשימה לשמירת תוצאות low עבור כל seed.
    low_runs = []

    # רשימה לשמירת תוצאות high עבור כל seed.
    high_runs = []

    # מעבר על כל seed ברשימת ההרצות.
    for seed in SEEDS:

        # שמירת זמן התחלה עבור seed נוכחי.
        start = time.time()

        # הרצת הניסוי עבור seed אחד וקבלת overlap במצב low ו-high.
        low_overlap, high_overlap = run_one_seed(seed)

        # שמירת תוצאות low של seed זה.
        low_runs.append(low_overlap)

        # שמירת תוצאות high של seed זה.
        high_runs.append(high_overlap)

        # הדפסת זמן הריצה של seed זה בדקות.
        print(f"\nSeed {seed} finished in {(time.time() - start) / 60:.2f} min")

    # חישוב ממוצע overlap של low על פני כל ה-seeds.
    low_avg = np.mean(np.stack(low_runs), axis=0)

    # חישוב ממוצע overlap של high על פני כל ה-seeds.
    high_avg = np.mean(np.stack(high_runs), axis=0)

    # הדפסת כותרת לתוצאות הממוצעות.
    print("\nAveraged layer-wise Fisher overlap:")

    # הדפסת ממוצע overlap עבור low.
    print(f"Low 8x8:    {low_avg.tolist()}")

    # הדפסת ממוצע overlap עבור high.
    print(f"High 26x26: {high_avg.tolist()}")

    # ציור הגרף הסופי של Figure C.
    plot_graph_c(low_avg, high_avg)

    # הדפסת זמן הריצה הכולל של כל הניסוי.
    print(f"\nTotal runtime: {(time.time() - total_start) / 60:.2f} min")
