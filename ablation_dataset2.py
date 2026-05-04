# dronepropb_ablation.py
# Ablation study for Dataset 2 (DronePropB) - comparing raw, RMS, and kurtosis inputs

import os
import time
import argparse
import random
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "Healthy",
    "Edge Cut SV1",
    "Edge Cut SV2",
    "Edge Cut SV3",
    "Crack SV1",
    "Crack SV2",
    "Crack SV3",
    "Surface Cut SV1",
    "Surface Cut SV2",
    "Surface Cut SV3",
]

FILE_LABEL = {
    "F0_SV0_SP2_C1_R1": 0,
    "F1_SV1_SP2_C1":    1,
    "F1_SV2_SP2_C1":    2,
    "F1_SV3_SP2_C1":    3,
    "F2_SV1_SP2_C1":    4,
    "F2_SV2_SP2_C1":    5,
    "F2_SV3_SP2_C1":    6,
    "F3_SV1_SP2_C1":    7,
    "F3_SV2_SP2_C1":    8,
    "F3_SV3_SP2_C1":    9,
}

BATCH_SIZE  = 64
WINDOW_SIZE = 2048


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_all_signals(data_dir: str, channel: str):
    data_dir = Path(data_dir)
    ch_idx   = {"X": 0, "Y": 1, "Z": 2}[channel.upper()]
    signals  = {}

    for fpath in sorted(data_dir.glob("*.mat")):
        stem = fpath.stem
        if stem not in FILE_LABEL:
            continue
        label = FILE_LABEL[stem]
        mat   = scipy.io.loadmat(str(fpath))
        imu   = mat["IMUData"][0, 0]
        sig   = imu["Vib_Data"][ch_idx, :].astype(np.float32)
        sig   = sig[np.isfinite(sig)]
        signals[label] = sig
        print(f"  Loaded {stem} -> {CLASS_NAMES[label]} ({len(sig)} samples)")

    return signals


def create_windows(signals: dict, window_size: int, method: str):
    all_samples, all_labels = [], []

    for label, sig in signals.items():
        n_windows = len(sig) // window_size
        for w in range(n_windows):
            start   = w * window_size
            segment = sig[start:start + window_size].astype(np.float32)

            if method == "raw":
                segment = (segment - segment.mean()) / (segment.std() + 1e-8)
            elif method == "rms":
                val     = np.sqrt(np.mean(segment ** 2) + 1e-12).astype(np.float32)
                segment = np.full_like(segment, val, dtype=np.float32)
            elif method == "kurtosis":
                val     = float(kurtosis(segment, fisher=False, bias=False))
                segment = np.full_like(segment, np.float32(val), dtype=np.float32)

            all_samples.append(segment)
            all_labels.append(label)

    return np.stack(all_samples), np.array(all_labels, dtype=np.int64)


class VibDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels  = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class CNN1D(nn.Module):
    def __init__(self, num_classes=10):
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

    total_loss, total_correct, total_n = 0.0, 0, 0

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
            out = model(x.to(device))
            preds.append(torch.argmax(out, 1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def append_csv(row: dict, out_csv: str):
    import pandas as pd
    df_row = pd.DataFrame([row])
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_new = df_row
    df_new.to_csv(out_csv, index=False)


def run_one(data_dir, method, channel, batch, window, epochs, lr, seed, out_csv=None, last_k=10):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading signals - axis={channel.upper()}")
    signals = load_all_signals(data_dir, channel)

    samples, labels = create_windows(signals, window, method)
    print(f"  Total windows: {len(samples)} | method={method} | window={window}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        samples, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    train_loader = DataLoader(VibDataset(X_train, y_train), batch_size=batch, shuffle=True)
    val_loader   = DataLoader(VibDataset(X_val,   y_val),   batch_size=batch, shuffle=False)
    test_loader  = DataLoader(VibDataset(X_test,  y_test),  batch_size=batch, shuffle=False)

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

    preds, true_labels = evaluate_preds(model, test_loader, device)
    p, r, f1, _        = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )

    k          = min(last_k, len(test_hist))
    last_val   = np.array(val_hist[-k:])
    last_test  = np.array(test_hist[-k:])
    val_mean,  val_std  = float(last_val.mean()),  float(last_val.std())
    test_mean, test_std = float(last_test.mean()), float(last_test.std())

    print("\n--- ABLATION METRICS ---")
    print(f"Method : {method.upper()} | Axis: {channel.upper()}")
    print(f"Val Accuracy  (last {k}): {val_mean*100:.2f} ± {val_std*100:.2f}%")
    print(f"Test Accuracy (last {k}): {test_mean*100:.2f} ± {test_std*100:.2f}%")
    print(f"Macro Precision : {p*100:.2f}%")
    print(f"Macro Recall    : {r*100:.2f}%")
    print(f"Macro F1-Score  : {f1*100:.2f}%")
    print(f"Training Time   : {train_time:.2f}s")

    if out_csv:
        row = {
            "config":  f"{method.upper()}-{channel.upper()}",
            "method":  method,
            "axis":    channel.upper(),
            "window":  int(window),
            "batch":   int(batch),
            "epochs":  int(epochs),
            "lr":      float(lr),
            "seed":    int(seed),
            f"val_acc_last{k}_mean":  round(val_mean  * 100, 2),
            f"val_acc_last{k}_std":   round(val_std   * 100, 2),
            f"test_acc_last{k}_mean": round(test_mean * 100, 2),
            f"test_acc_last{k}_std":  round(test_std  * 100, 2),
            "macro_precision": round(float(p),  4),
            "macro_recall":    round(float(r),  4),
            "macro_f1":        round(float(f1), 4),
            "train_time_s":    round(float(train_time), 2),
        }
        append_csv(row, out_csv)
        print(f"Saved row to: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,   default="data")
    parser.add_argument("--method",   type=str,   default="raw",  choices=["raw", "rms", "kurtosis"])
    parser.add_argument("--channel",  type=str,   default="Z",    choices=["X", "Y", "Z"])
    parser.add_argument("--batch",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--window",   type=int,   default=WINDOW_SIZE)
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--last_k",   type=int,   default=10)
    parser.add_argument("--run_all",  action="store_true")
    parser.add_argument("--out_csv",  type=str,   default="dronepropb_ablation_results.csv")
    parser.add_argument("--no_csv",   action="store_true")

    args    = parser.parse_args()
    out_csv = None if args.no_csv else args.out_csv

    if args.run_all:
        configs = [
            ("raw",      "X"), ("raw",      "Y"), ("raw",      "Z"),
            ("rms",      "X"), ("rms",      "Y"), ("rms",      "Z"),
            ("kurtosis", "X"), ("kurtosis", "Y"), ("kurtosis", "Z"),
        ]
        for idx, (m, ch) in enumerate(configs, start=1):
            print("\n" + "=" * 80)
            print(f"CONFIG A{idx} | method={m} | axis={ch} | batch={args.batch} | "
                  f"window={args.window} | epochs={args.epochs}")
            print("=" * 80)
            run_one(
                args.data_dir, m, ch,
                args.batch, args.window, args.epochs, args.lr,
                args.seed, out_csv, last_k=args.last_k
            )
        print("\n" + "=" * 80)
        print("ALL 9 CONFIGURATIONS DONE")
        if out_csv:
            print(f"Full results saved to: {out_csv}")
        print("=" * 80)
    else:
        run_one(
            args.data_dir, args.method, args.channel,
            args.batch, args.window, args.epochs, args.lr,
            args.seed, out_csv, last_k=args.last_k
        )


if __name__ == "__main__":
    main()
