from sklearn.metrics import recall_score
from torch.utils.data import DataLoader

import torch
from torch import nn, optim
from tqdm import tqdm

from jett.dataset.affectnet_mediapipe import FileLoader, FaceDataset
from jett.modules import JETT, JETTBuilder
from jett.train import JETTConfigs, JETTTrainParams, Metrics, JETTTrainer
from jett.dataset.affectnet_mediapipe import NUM_POINTS

# Training Hyperparam
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60

# Model Architecture
DIM_MODEL = 256
NUM_HEADS = 4
NUM_CLASSES = 8
NUM_LAYERS = 5
NEAREST_K = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AffectNetMetrics(Metrics):

    def print(self):
        loss, acc, uar = self.values
        return f"{self.identity}: loss={loss:.2f}, acc={acc:.2f}%, uar={uar:.2f}%"


def calculate_uar(y_true, y_pred, num_classes):
    """
    Calculate UAR
    """
    return recall_score(y_true, y_pred, average='macro', zero_division=0) * 100


def train_one_epoch(_model: JETT, _train_params: JETTTrainParams):
    _model.train()
    tot_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(_train_params.train_loader, desc="Training")
    for coords, labels in pbar:
        coords, labels = coords.to(_train_params.device), labels.to(_train_params.device)

        outputs = _model(coords)
        loss = _train_params.criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

        # Get prediction from one-hot
        _, pred = torch.max(outputs, dim=1)  # First dim is batch size
        correct += torch.eq(pred, labels).sum().item()
        total += labels.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        avg_loss = tot_loss / (total / BATCH_SIZE)
        avg_acc = 100 * correct / total
        uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%", "uar": f"{uar:.2f}%"})

    avg_loss = tot_loss / len(train_loader)
    avg_acc = 100 * correct / total
    uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
    return AffectNetMetrics(
        identity="Train", keys=["loss", "accuracy", "UAR"],
        values=[avg_loss, avg_acc, uar], primary_key=2)


def validate_one_epoch(_model: JETT, _train_params: JETTTrainParams):
    _model.eval()
    tot_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(_train_params.valid_loader, desc="Validating")
        for coords, labels in pbar:
            coords, labels = coords.to(_train_params.device), labels.to(_train_params.device)

            outputs = _model(coords)
            loss = _train_params.criterion(outputs, labels)

            tot_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.eq(pred, labels).sum().item()
            total += labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            avg_loss = tot_loss / (total / BATCH_SIZE)
            avg_acc = 100 * correct / total
            uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
            pbar.set_postfix({"val_loss": f"{avg_loss:.4f}", "val_acc": f"{avg_acc:.2f}%", "val_uar": f"{uar:.2f}%"})

    avg_loss = tot_loss / len(valid_loader)
    avg_acc = 100 * correct / total
    uar = calculate_uar(all_labels, all_preds, NUM_CLASSES)
    return AffectNetMetrics(
        identity="Valid", keys=["loss", "accuracy", "UAR"],
        values=[avg_loss, avg_acc, uar], primary_key=2)


if __name__ == "__main__":
    # Prepare Dataset
    file_loader = FileLoader()
    train_data, train_labels = file_loader.load_split("Train")
    valid_data, valid_labels = file_loader.load_split("Test")

    train_dataset = FaceDataset(train_data, train_labels)
    valid_dataset = FaceDataset(valid_data, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build up model structure
    classifier = nn.Linear(NUM_POINTS * DIM_MODEL, NUM_CLASSES)

    model = JETTBuilder().dim_model(DIM_MODEL) \
        .num_heads(NUM_HEADS) \
        .nearest_k(NEAREST_K) \
        .out_module(classifier) \
        .num_layers(NUM_LAYERS) \
        .drop_rate(0.1) \
        .build()

    # Configure Training
    configs = JETTConfigs(epochs=NUM_EPOCHS)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.epochs)

    train_params = JETTTrainParams(
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE)

    trainer = JETTTrainer(
        model=model,
        save_target="models/affectnet_jett",
        configs=configs,
        train_params=train_params,
        train_one_epoch=train_one_epoch,
        valid_one_epoch=validate_one_epoch,
        is_better=lambda best, cur_uar: cur_uar > best)

    trainer.train().print_result()
