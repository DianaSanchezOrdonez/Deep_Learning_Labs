import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image

# ============================
# 1. Configuration and Constants
# ============================
# ImageNet normalization values
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Paths and configuration
images_dir = "images"
train_csv = "train.csv"
eval_csv = "eval.csv"
test_csv = "test.csv"

# Training parameters
n_batch = 64
n_workers = 2
n_epochs = 20
img_size = (224, 224)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = True if torch.cuda.is_available() else False
print(f"Using device: {device}")

# ============================
# 2. Custom Dataset Class
# ============================
class AvionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.csv_file = csv_file

        # Read CSV file
        df = pd.read_csv(self.csv_file)
        df["filename"] = df["filename"].astype(str)
        df["Labels"] = df["Labels"].astype(int)
        self.df = df.reset_index(drop=True)

        # Transforms/data augmentation
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        y = int(row["Labels"])
        return {"image": img, "label": y}

# ============================
# 3. Data Transforms
# ============================
trainTransforms = transforms.Compose([
    # Random cropping and resizing for data augmentation
    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),

    # Advanced augmentation block applied with 60% probability
    transforms.RandomApply([
        transforms.RandomChoice([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.RandomAdjustSharpness(1.5),
        ]),
        transforms.RandomGrayscale(p=0.10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.30),
    ], p=0.60),

    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

evalTransforms = transforms.Compose([
    transforms.Resize(256),  # Resize shorter side to 256
    transforms.CenterCrop(img_size),  # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ============================
# 4. Dataset and DataLoader Creation
# ============================
trainDataset = AvionDataset(train_csv, images_dir, transform=trainTransforms)
evalDataset = AvionDataset(eval_csv, images_dir, transform=evalTransforms)
testDataset = AvionDataset(test_csv, images_dir, transform=evalTransforms)

trainDataLoader = DataLoader(
    trainDataset,
    batch_size=n_batch,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=pin_memory,
    persistent_workers=True if n_workers > 0 else False
)

evalDataLoader = DataLoader(
    evalDataset,
    batch_size=n_batch,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=pin_memory,
    persistent_workers=True if n_workers > 0 else False
)

testDataLoader = DataLoader(
    testDataset,
    batch_size=n_batch,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=pin_memory,
    persistent_workers=True if n_workers > 0 else False
)

print(f"Dataset sizes - Train: {len(trainDataset)}, Eval: {len(evalDataset)}, Test: {len(testDataset)}")

# ============================
# 5. Model Setup
# ============================
model = resnet18(weights='IMAGENET1K_V1')

# OPCIÓN 1: Fine-tuning completo (actual) - todas las capas se entrenan
freeze_backbone = True # Cambiar a True para congelar

if freeze_backbone:
    # OPCIÓN 2: Congelar todas las capas excepto la final (feature extraction)
    for param in model.parameters():
        param.requires_grad = False
    print("Backbone congelado - solo se entrenará la capa clasificadora")
else:
    print("Fine-tuning completo - todas las capas se entrenarán")

# Reemplazar capa clasificadora
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 100)  # Assuming 100 aircraft classes

# Si congelamos, asegurar que la nueva capa final sea entrenable
if freeze_backbone:
    for param in model.fc.parameters():
        param.requires_grad = True

model = model.to(device)

# Contar parámetros entrenables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros totales: {total_params:,}")
print(f"Parámetros entrenables: {trainable_params:,}")
print(f"Parámetros congelados: {total_params - trainable_params:,}")

# ============================
# 6. Loss Function and Optimizer
# ============================
criterion = nn.CrossEntropyLoss()

# ESTRATEGIA DE OPTIMIZACIÓN
if freeze_backbone:
    # Solo optimizar la capa clasificadora
    optimizer = Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    print("Optimizando solo la capa clasificadora")
else:
    # Fine-tuning con diferentes learning rates
    # Backbone con LR menor, clasificador con LR mayor
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = Adam([
        {'params': backbone_params, 'lr': 1e-4},      # LR menor para backbone
        {'params': classifier_params, 'lr': 1e-3}     # LR mayor para clasificador
    ], weight_decay=1e-4)
    print("Optimización con learning rates diferenciados")

# ============================
# 7. Training and Evaluation Functions
# ============================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, data in enumerate(dataloader):
        inputs = data['image'].to(device, non_blocking=True)
        labels = data['label'].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}')

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data in dataloader:
            inputs = data['image'].to(device, non_blocking=True)
            labels = data['label'].to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================
# 8. Training Loop with Metrics Tracking
# ============================
history = {
    "train_loss": [],
    "train_acc": [],
    "eval_loss": [],
    "eval_acc": [],
    "test_loss": [],
    "test_acc": []
}

print("Starting training...")
best_eval_acc = 0.0

for epoch in range(n_epochs):
    print(f"\nEpoch [{epoch+1}/{n_epochs}]")

    train_loss, train_acc = train_one_epoch(model, trainDataLoader, criterion, optimizer, device)
    eval_loss, eval_acc = evaluate(model, evalDataLoader, criterion, device)

    # Save metrics
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["eval_loss"].append(eval_loss)
    history["eval_acc"].append(eval_acc)

    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Eval Loss: {eval_loss:.4f} Acc: {eval_acc:.2f}%")

    # Save best model
    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved with eval accuracy: {best_eval_acc:.2f}%")

# ============================
# 9. Final Test Evaluation
# ============================
print("\nEvaluating on test set...")
test_loss, test_acc = evaluate(model, testDataLoader, criterion, device)
history["test_loss"].append(test_loss)
history["test_acc"].append(test_acc)

print(f"Final Test Results - Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

# ============================
# 10. Plot Training Metrics
# ============================
epochs = range(1, n_epochs + 1)

plt.figure(figsize=(15, 5))

# Loss subplot
plt.subplot(1, 3, 1)
plt.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
plt.plot(epochs, history["eval_loss"], 'r-', label="Eval Loss", linewidth=2)
plt.axhline(y=test_loss, color="green", linestyle="--", label=f"Test Loss ({test_loss:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy subplot
plt.subplot(1, 3, 2)
plt.plot(epochs, history["train_acc"], 'b-', label="Train Acc", linewidth=2)
plt.plot(epochs, history["eval_acc"], 'r-', label="Eval Acc", linewidth=2)
plt.axhline(y=test_acc, color="green", linestyle="--", label=f"Test Acc ({test_acc:.1f}%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.grid(True, alpha=0.3)

# Training summary
plt.subplot(1, 3, 3)
plt.axis('off')
summary_text = f"""Training Summary:
────────────────────
Epochs: {n_epochs}
Best Eval Acc: {best_eval_acc:.2f}%
Final Test Acc: {test_acc:.2f}%
Final Test Loss: {test_loss:.4f}

Dataset Sizes:
Train: {len(trainDataset)}
Eval: {len(evalDataset)}
Test: {len(testDataset)}

Model: ResNet-18
Classes: 100
Device: {device}
"""
plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', fontfamily='monospace')

plt.tight_layout()
plt.show()

# ============================
# 11. Save Training History
# ============================
import json
with open('training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\nTraining completed!")
print(f"Best model saved as 'best_model.pth'")
print(f"Training history saved as 'training_history.json'")