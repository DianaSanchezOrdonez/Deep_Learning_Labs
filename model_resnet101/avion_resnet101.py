import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet101
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import numpy as np

# ============================
# Configuración principal
# ============================
freeze_backbone = False  # Cambia a True si quieres solo feature extraction
nombre_modelo = "best_model_resnet101_flush.pth"  # Nombre del modelo guardado
nombre_imagen = "training_history_resnet101.png"
historial_entrenamiento = "training_history_resnet101.json"  # Archivo para guardar el historial de entrenamiento

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

images_dir = "images"
train_csv = "new_train.csv"
eval_csv = "new_val.csv"
test_csv = "new_test.csv"

n_batch = 64        # Batch size fijo en 64
n_workers = 8
n_epochs = 100
img_size = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = True if torch.cuda.is_available() else False
print(f"Using device: {device}", flush=True)

# ============================
# Dataset personalizado
# ============================
class AvionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        df = pd.read_csv(csv_file)
        df["filename"] = df["filename"].astype(str)
        df["Labels"] = df["Labels"].astype(int)
        self.df = df.reset_index(drop=True)
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
# Data augmentation
# ============================

trainTransforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.7, 1.4)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(
        degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10
    ),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.RandomAdjustSharpness(2.0),
    ], p=0.9),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

evalTransforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ============================
# DataLoaders
# ============================
trainDataset = AvionDataset(train_csv, images_dir, transform=trainTransforms)
evalDataset = AvionDataset(eval_csv, images_dir, transform=evalTransforms)
testDataset = AvionDataset(test_csv, images_dir, transform=evalTransforms)

trainDataLoader = DataLoader(trainDataset, batch_size=n_batch, shuffle=True,
                             num_workers=n_workers, pin_memory=pin_memory,
                             persistent_workers=True if n_workers > 0 else False)

evalDataLoader = DataLoader(evalDataset, batch_size=n_batch, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_memory,
                            persistent_workers=True if n_workers > 0 else False)

testDataLoader = DataLoader(testDataset, batch_size=n_batch, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_memory,
                            persistent_workers=True if n_workers > 0 else False)

print(f"Dataset sizes - Train: {len(trainDataset)}, Eval: {len(evalDataset)}, Test: {len(testDataset)}", flush=True)

# ============================
# Modelo ResNet-18
# ============================
model = resnet101(weights=None)
state_dict = torch.load('resnet101-63fe2227.pth', map_location='cpu')
model.load_state_dict(state_dict)
if freeze_backbone:
    for param in model.parameters():
        param.requires_grad = False
    print("Backbone congelado - solo se entrenará la capa clasificadora", flush=True)
else:
    print("Fine-tuning completo - todas las capas se entrenarán", flush=True)

# Reemplazar capa clasificadora
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 100)  # 100 clases

if freeze_backbone:
    for param in model.fc.parameters():
        param.requires_grad = True

model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros totales: {total_params:,}", flush=True)
print(f"Parámetros entrenables: {trainable_params:,}", flush=True)
print(f"Parámetros congelados: {total_params - trainable_params:,}", flush=True)

# ============================
# Loss & Optimizer
# ============================
criterion = nn.CrossEntropyLoss()

if freeze_backbone:
    optimizer = Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    print("Optimizando solo la capa clasificadora", flush=True)
else:
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = Adam([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': classifier_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    print("Optimización con learning rates diferenciados", flush=True)

# ============================
# Early Stopping
# ============================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.25):
        self.patience = patience
        self.min_delta = min_delta
        self.best_acc = 0.0
        self.counter = 0
        self.early_stop = False

    def __call__(self, eval_acc):
        if eval_acc > self.best_acc + self.min_delta:
            self.best_acc = eval_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ============================
# Funciones de entrenamiento y evaluación
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

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

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

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    # Calcular métricas adicionales
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    return epoch_loss, epoch_acc, precision, recall, f1

# ============================
# Bucle de entrenamiento con Early Stopping
# ============================
history = {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": [], "test_loss": [], "test_acc": []}
early_stopper = EarlyStopping(patience=15, min_delta=0.1)
best_eval_acc = 0.0

print("Starting training...", flush=True)

for epoch in range(n_epochs):
    print(f"\nEpoch [{epoch+1}/{n_epochs}]", flush=True)

    train_loss, train_acc = train_one_epoch(model, trainDataLoader, criterion, optimizer, device)
    eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = evaluate(model, evalDataLoader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["eval_loss"].append(eval_loss)
    history["eval_acc"].append(eval_acc)
    history.setdefault("eval_precision", []).append(eval_precision)
    history.setdefault("eval_recall", []).append(eval_recall)
    history.setdefault("eval_f1", []).append(eval_f1)

    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
      f"Eval Loss: {eval_loss:.4f} Acc: {eval_acc:.2f}% "
      f"Prec: {eval_precision:.2f} Recall: {eval_recall:.2f} F1: {eval_f1:.2f}", flush=True)
    
    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        torch.save(model.state_dict(), nombre_modelo)
        print(f"✅ New best model saved with eval accuracy: {best_eval_acc:.2f}%", flush=True)

    # Early stopping
    early_stopper(eval_acc)
    if early_stopper.early_stop:
        print(f"\n⏹️ Early stopping activado en epoch {epoch+1}", flush=True)
        break

# ============================
# Evaluación final en test
# ============================
print("\nEvaluating on test set...", flush=True)
test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, testDataLoader, criterion, device)
history["test_loss"].append(test_loss)
history["test_acc"].append(test_acc)
history["test_precision"] = test_precision
history["test_recall"] = test_recall
history["test_f1"] = test_f1

print(f"Final Test Results - Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% "
      f"| Prec: {test_precision:.2f} | Recall: {test_recall:.2f} | F1: {test_f1:.2f}", flush=True)


# ============================
# Guardar métricas
# ============================
with open(historial_entrenamiento, 'w') as f:
    json.dump(history, f, indent=2)

print("\nTraining completed!", flush=True)
print(f"Best model saved as '{nombre_modelo}'", flush=True)
print(f"Training history saved as 'training_history.json'", flush=True)

# ============================
# Guardar gráfico de entrenamiento
# ============================
def save_training_plot(history, filename="training_plot.png"):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Loss ---
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:red", linestyle="--")
    ax1.plot(epochs, history["eval_loss"], label="Eval Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # --- Accuracy ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color="tab:blue")
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="tab:blue", linestyle="--")
    ax2.plot(epochs, history["eval_acc"], label="Eval Acc", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # --- Leyenda combinada ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")

    plt.title("Training & Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # alta resolución
    plt.close(fig)  # cerrar figura para liberar memoria

# Guardar gráfico
save_training_plot(history, nombre_imagen)
print("📊 Gráfico guardado como 'training_history_resnet101.png'", flush=True)