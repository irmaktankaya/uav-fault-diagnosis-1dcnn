# main_dataset1.py
# 1D CNN for UAV rotor blade fault diagnosis - Dataset 1


import os
import time
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

import matplotlib.pyplot as plt

CLASS_NAMES = [
    "Healthy",
    "Damaged Bottom Right Blade",
    "Damaged Top Right Blade",
    "Unbalanced Bottom Right Blade",
    "Unbalanced Top Right Blade",
]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
WINDOW_SIZE = 4096  # change input length to 512, 1024, 2048, 4096


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_label_from_filename(filename: str) -> int:
    for class_name, idx in LABEL_MAP.items():
        if class_name in filename:
            return idx


def load_signal_by_index(path: Path, channel: str) -> np.ndarray:
    df = pd.read_excel(path, sheet_name=0, header=0, engine="openpyxl")
    data = df.to_numpy()
    col = {"x": 1, "y": 2, "z": 3}[channel.strip().lower()]
    sig = data[:, col].astype(np.float32)
    return sig[np.isfinite(sig)]


def save_confusion_matrix_figure(y_true, y_pred, class_names, out_basepath,
                                  normalize=True, show_counts=True, title=None, dpi_png=600):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_plot = cm.astype(float)

    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums

    fig_w = max(7.0, 0.95 * len(class_names) + 3.0)
    fig_h = max(6.0, 0.85 * len(class_names) + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm_plot, interpolation="nearest")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    if title is None:
        title = "Confusion Matrix" + (" (Normalized)" if normalize else "")
    ax.set_title(title, fontsize=13, pad=10)

    thresh = cm_plot.max() * 0.65
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                pct = cm_plot[i, j] * 100.0
                text = f"{pct:.1f}%\n(n={cm[i, j]})" if show_counts else f"{pct:.1f}%"
            else:
                text = f"{cm[i, j]}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if cm_plot[i, j] > thresh else "black")

    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    ax.grid(False)
    fig.tight_layout()

    pdf_path = out_basepath + ".pdf"
    png_path = out_basepath + ".png"
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)

    return cm


class DroneBladeDataset(Dataset):
    def __init__(self, root_dir="ML_data", split="train", channel="X", use_fft=False,
                 window_size=WINDOW_SIZE):
        self.root_dir = Path(root_dir)
        self.split = split
        self.channel = channel
        self.use_fft = use_fft
        self.window_size = window_size

        split_dir = self.root_dir / split
        excel_files = sorted(split_dir.glob("*.xlsx"))

        print(f"[{split}] Loading {len(excel_files)} files from {split_dir}")
        print(f"[{split}] Channel={self.channel} | FFT={self.use_fft} | Window={self.window_size}")

        self.samples = []
        self.labels = []

        for file in excel_files:
            label = infer_label_from_filename(file.name)
            signal = load_signal_by_index(file, self.channel)
            n = len(signal)
            num_windows = n // self.window_size

            for w in range(num_windows):
                start = w * self.window_size
                end = start + self.window_size
                segment = signal[start:end]

                if self.use_fft:
                    segment = np.abs(np.fft.fft(segment))

                self.samples.append(segment.astype(np.float32))
                self.labels.append(label)

        self.samples = np.stack(self.samples)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"[{split}] Created {len(self.samples)} windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def accuracy(logits, labels):
    return (torch.argmax(logits, 1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, opt, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            opt.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            acc = accuracy(out, y)

        if train:
            loss.backward()
            opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, 1).cpu().numpy()
            preds.append(pred)
            labels.append(y.numpy())

    return np.concatenate(preds), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default="X")
    parser.add_argument("--use_fft", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device = {device}")

    train_ds = DroneBladeDataset("ML_data", "train", args.channel, args.use_fft)
    val_ds   = DroneBladeDataset("ML_data", "validate", args.channel, args.use_fft)
    test_ds  = DroneBladeDataset("ML_data", "test", args.channel, args.use_fft)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model     = CNN1D(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=1e-3)

    test_hist  = []
    best_val   = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss,  tr_acc  = run_epoch(model, train_loader, criterion, opt, device, True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion, opt, device, False)
        _,        test_acc = run_epoch(model, test_loader,  criterion, opt, device, False)

        test_hist.append(test_acc)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {ep:03d} | "
            f"Train {tr_acc*100:6.2f}% | "
            f"Val {val_acc*100:6.2f}% | "
            f"Test {test_acc*100:6.2f}% | "
            f"Time {time.time()-t0:.2f}s"
        )

    print(f"\nBest val accuracy: {best_val*100:.2f}%")

    model.load_state_dict(best_state)
    model.to(device)

    preds, labels = evaluate(model, test_loader, device)
    final_acc = (preds == labels).mean()
    print(f"\nFinal Test Accuracy: {final_acc*100:.2f}%")

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    print(f"\nMacro Precision: {p*100:.2f}%")
    print(f"Macro Recall:    {r*100:.2f}%")
    print(f"Macro F1:        {f1*100:.2f}%")

    print("\nPer-Class Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    print("\nConfusion Matrix (raw counts):")
    print(cm)

    mode = "fft" if args.use_fft else "time"
    ch   = args.channel.upper()
    os.makedirs(args.outdir, exist_ok=True)

    out_base = os.path.join(args.outdir, f"confusion_matrix_{ch}_{mode}_journal")
    save_confusion_matrix_figure(
        y_true=labels,
        y_pred=preds,
        class_names=CLASS_NAMES,
        out_basepath=out_base,
        normalize=True,
        show_counts=True,
        title=f"Confusion Matrix ({ch}, {mode})",
        dpi_png=600,
    )
    print(f"\nSaved the confusion matrix")

    last_vals = np.array(test_hist[-10:]) if len(test_hist) >= 10 else np.array(test_hist)
    print(
        f"\nTest Accuracy (Last {len(last_vals)} Epochs): "
        f"{last_vals.mean()*100:.2f} ± {last_vals.std()*100:.2f} %"
    )


if __name__ == "__main__":
    main()
