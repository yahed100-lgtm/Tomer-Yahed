import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# הגדרת המכשיר (כרטיס מסך או מעבד)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. הכנת הנתונים
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# הורדת MNIST המקורי
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def get_split_mnist(dataset, digits):
    """פונקציה המסננת רק את הספרות שבחרנו עבור המשימה"""
    indices = [i for i, label in enumerate(dataset.targets) if label in digits]
    return Subset(dataset, indices)

# פיצול למשימות לפי המאמר
task1_train = DataLoader(get_split_mnist(full_train, [0, 1]), batch_size=64, shuffle=True)
task1_test = DataLoader(get_split_mnist(full_test, [0, 1]), batch_size=64, shuffle=False)

task2_train = DataLoader(get_split_mnist(full_train, [2, 3]), batch_size=64, shuffle=True)
task2_test = DataLoader(get_split_mnist(full_test, [2, 3]), batch_size=64, shuffle=False)

print(f"נתוני המשימות מוכנים על המכשיר: {device}")

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 10) # 10 יציאות עבור כל הספרות (0-9)
        )

    def forward(self, x):
        return self.fc(x)

model = SimpleMLP().to(device)
print("המודל נבנה בהצלחה.")


# פונקציה לבדיקת דיוק המודל (Accuracy)
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# הגדרת פונקציית הפסד ואופטימייזר (לפי המאמר)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# --- שלב א': אימון על משימה 1 ---
print("\nTraining on Task 1 (0, 1)...")
for epoch in range(2):
    model.train()
    for images, labels in task1_train:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    acc1 = evaluate(model, task1_test)
    print(f"Epoch {epoch + 1}: Accuracy on Task 1: {acc1:.2f}%")

# --- שלב ב': אימון על משימה 2 ---
print("\nTraining on Task 2 (2, 3)...")
for epoch in range(2):
    model.train()
    for images, labels in task2_train:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    acc2 = evaluate(model, task2_test)
    print(f"Epoch {epoch + 1}: Accuracy on Task 2: {acc2:.2f}%")

# --- הבדיקה הסופית: האם הוא שכח? ---
print("\n--- Final Results (Catastrophic Forgetting Check) ---")
final_acc1 = evaluate(model, task1_test)
print(f"Final Accuracy on Task 1 (0, 1): {final_acc1:.2f}%")
print(f"Final Accuracy on Task 2 (2, 3): {acc2:.2f}%")