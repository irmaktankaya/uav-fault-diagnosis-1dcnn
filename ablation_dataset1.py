# ablation.py
# Ablation study for Dataset 1 - comparing raw, RMS, and kurtosis input representations


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
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import kurtosis

CLASS_NAMES = [
    "Healthy",
    "Damaged Bottom Right Blade",
    "Damaged Top Right Blade",
    "Unbalanced Bottom Right Blade",
    "Unbalanced Top Right Blade",
]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


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


class DroneBladeDataset(Dataset):
    def __init__(self, root_dir, split, channel, window_size, method="raw"):
        self.root_dir    = Path(root_dir)
        self.split       = split
        self.channel     = channel
        self.window_size = int(window_size)
        self.method      = method

        split_dir    = self.root_dir / split
        excel_files  = sorted(split_dir.glob("*.xlsx"))

        self.samples = []
        self.labels  = []

        for file in excel_files:
            label   = infer_label_from_filename(file.name)
            signal  = load_signal_by_index(file, self.channel)

            num_windows = len(signal) // self.window_size
            for w in range(num_windows):
                start   = w * self.window_size
                segment = signal[start:start + self.window_size].astype(np.float32)

                if self.method == "rms":
                    val     = np.sqrt(np.mean(segment ** 2) + 1e-12).astype(np.float32)
                    segment = np.full_like(segment, val, dtype=np.float32)
                elif self.method == "kurtosis":
                    val     = float(kurtosis(segment, fisher=False, bias=False))
                    segment = np.full_like(segment, np.float32(val), dtype=np.float32)
                elif self.method == "raw":
                    segment = (segment - segment.mean()) / (segment.std() + 1e-8)

                self.samples.append(segment)
                self.labels.append(label)

        self.samples = np.stack(self.samples)
        self.labels  = np.array(self.labels, dtype=np.int64)

        print(f"[{split}] windows={len(self.samples)} | method={self.method} | "
              f"axis={self.channel} | window={self.window_size}")

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


def run_epoch(model, loader, criterion, opt, device, train=True):
    model.train() if train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_n       = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train and opt is not None:
            opt.zero_grad()

        with torch.set_grad_enabled(train):
            out  = model(x)
            loss = criterion(out, y)

        if train and opt is not None:
            loss.backward()
            opt.step()

        total_loss    += loss.item() * x.size(0)
        total_correct += (torch.argmax(out, 1) == y).sum().item()
        total_n       += x.size(0)

    return total_loss / total_n, total_correct / total_n


def evaluate_preds(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.append(torch.argmax(out, 1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def append_csv(row: dict, out_csv: str):
    df_row = pd.DataFrame([row])
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_new = df_row
    df_new.to_csv(out_csv, index=False)


def run_one(root, method, channel, window, batch, epochs, lr, seed, out_csv=None, last_k=10):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DroneBladeDataset(root, "train",    channel, window, method)
    val_ds   = DroneBladeDataset(root, "validate", channel, window, method)
    test_ds  = DroneBladeDataset(root, "test",     channel, window, method)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    model     = CNN1D(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=lr)

    best_val   = -1.0
    best_state = None
    val_hist   = []
    test_hist  = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        _, tr_acc  = run_epoch(model, train_loader, criterion, opt,  device, train=True)
        _, val_acc = run_epoch(model, val_loader,   criterion, None, device, train=False)
        _, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)

        val_hist.append(val_acc)
        test_hist.append(test_acc)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{method.upper()}-{channel.upper()}] Epoch {ep:02d} | "
              f"Train {tr_acc*100:.2f}% | Val {val_acc*100:.2f}% | Test {test_acc*100:.2f}%")

    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.to(device)

    preds, labels = evaluate_preds(model, test_loader, device)
    p, r, f1, _   = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)

    k           = min(last_k, len(test_hist))
    last_val    = np.array(val_hist[-k:])
    last_test   = np.array(test_hist[-k:])
    val_mean,  val_std  = float(last_val.mean()),  float(last_val.std())
    test_mean, test_std = float(last_test.mean()), float(last_test.std())

    print("\n--- TABLE 3 METRICS ---")
    print(f"Method: {method.upper()} | Axis: {channel.upper()}")
    print(f"Validation Accuracy (last {k}): {val_mean*100:.2f} ± {val_std*100:.2f}")
    print(f"Test Accuracy       (last {k}): {test_mean*100:.2f} ± {test_std*100:.2f}")
    print(f"Macro Precision:  {p*100:.2f}%")
    print(f"Macro Recall:     {r*100:.2f}%")
    print(f"Macro F1-Score:   {f1*100:.2f}%")
    print(f"Training Time (s): {train_time:.2f}")

    if out_csv:
        row = {
            "model":   "CNN1D",
            "method":  method,
            "axis":    channel.upper(),
            "window":  int(window),
            "batch":   int(batch),
            "epochs":  int(epochs),
            "lr":      float(lr),
            "seed":    int(seed),
            f"val_acc_last{k}_mean":  val_mean,
            f"val_acc_last{k}_std":   val_std,
            f"test_acc_last{k}_mean": test_mean,
            f"test_acc_last{k}_std":  test_std,
            "macro_precision": float(p),
            "macro_recall":    float(r),
            "macro_f1":        float(f1),
            "train_time_s":    float(train_time),
        }
        append_csv(row, out_csv)
        print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     type=str,   default="ML_data")
    parser.add_argument("--window",   type=int,   default=4096)
    parser.add_argument("--batch",    type=int,   default=128)
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--last_k",   type=int,   default=10)
    parser.add_argument("--method",   type=str,   default="raw",   choices=["raw", "rms", "kurtosis"])
    parser.add_argument("--channel",  type=str,   default="Z",     choices=["X", "Y", "Z"])
    parser.add_argument("--run_all",  action="store_true")
    parser.add_argument("--out_csv",  type=str,   default="table3_results.csv")
    parser.add_argument("--no_csv",   action="store_true")

    args    = parser.parse_args()
    out_csv = None if args.no_csv else args.out_csv

    if args.run_all:
        for m in ["raw", "rms", "kurtosis"]:
            for ch in ["X", "Y", "Z"]:
                print("\n" + "=" * 90)
                print(f"RUN: method={m} | axis={ch} | window={args.window} | "
                      f"batch={args.batch} | epochs={args.epochs}")
                print("=" * 90)
                run_one(args.root, m, ch, args.window, args.batch,
                        args.epochs, args.lr, args.seed, out_csv, last_k=args.last_k)
    else:
        run_one(args.root, args.method, args.channel, args.window, args.batch,
                args.epochs, args.lr, args.seed, out_csv, last_k=args.last_k)


if __name__ == "__main__":
    main()
