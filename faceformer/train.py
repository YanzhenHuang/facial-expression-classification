import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

from faceformer import DEVICE, NUM_CLASSES
from faceformer.dataset import FileLoader, FaceDataset
from faceformer.modules import FaceFormer

# 最优超参数（适配你的模型）
BATCH_SIZE = 16
LR = 5e-4        # 比1e-4收敛快10倍
WEIGHT_DECAY = 1e-4
EPOCHS = 60

# ... existing code ...

def calculate_uar(y_true, y_pred, num_classes):
    """
    Calculate UAR
    """
    return recall_score(y_true, y_pred, average='macro', zero_division=0) * 100


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training")
    for coords, labels in pbar:
        coords, labels = coords.to(device), labels.to(device)

        outputs = model(coords)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / (total / BATCH_SIZE)
        avg_acc = 100 * correct / total
        uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%", "uar": f"{uar:.2f}%"})

    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total
    uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
    return avg_loss, avg_acc, uar


def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for coords, labels in pbar:
            coords, labels = coords.to(device), labels.to(device)

            outputs = model(coords)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            avg_loss = total_loss / (total / BATCH_SIZE)
            avg_acc = 100 * correct / total
            uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
            pbar.set_postfix({"val_loss": f"{avg_loss:.4f}", "val_acc": f"{avg_acc:.2f}%", "val_uar": f"{uar:.2f}%"})

    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total
    uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
    return avg_loss, avg_acc, uar


if __name__ == "__main__":
    print("加载训练数据...")
    file_loader = FileLoader()
    train_data, train_labels = file_loader.load_split("Train")
    val_data, val_labels = file_loader.load_split("Test")

    train_dataset = FaceDataset(train_data, train_labels)
    val_dataset = FaceDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FaceFormer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_uar = 0.0
    print(f"训练开始！设备：{DEVICE}，总类别：{NUM_CLASSES}")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        train_loss, train_acc, train_uar = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_uar = val_one_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if val_uar > best_uar:
            best_uar = val_uar
            torch.save(model.state_dict(), "../models/best_faceformer.pth")
            print(f"✅ 最佳模型已保存！最佳验证 UAR：{best_uar:.2f}%")

        print(
            f"Epoch {epoch} | "
            f"训练Loss: {train_loss:.4f} 训练Acc: {train_acc:.2f}% 训练UAR: {train_uar:.2f}% | "
            f"验证Loss: {val_loss:.4f} 验证Acc: {val_acc:.2f}% 验证UAR: {val_uar:.2f}%"
        )

    print("\n训练完成！最佳验证 UAR：{:.2f}%".format(best_uar))
