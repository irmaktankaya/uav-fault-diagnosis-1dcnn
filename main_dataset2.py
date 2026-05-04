# main_dataset2.py
# 1D CNN for UAV rotor blade fault diagnosis - Dataset 2

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
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

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

BATCH_SIZES  = [32, 64, 128, 256]
WINDOW_SIZES = [512, 1024, 2048, 4096]


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


def create_windows(signals: dict, window_size: int):
    all_samples, all_labels = [], []

    for label, sig in signals.items():
        n_windows = len(sig) // window_size
        for w in range(n_windows):
            start = w * window_size
            seg   = sig[start:start + window_size]
            all_samples.append(seg)
            all_labels.append(label)

    return np.stack(all_samples), np.array(all_labels, dtype=np.int64)


class VibDataset(Dataset):
    def __init__(self, samples, labels, use_fft=False):
        self.samples = samples
        self.labels  = labels
        self.use_fft = use_fft

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg = self.samples[idx]
        if self.use_fft:
            seg = np.abs(np.fft.fft(seg)).astype(np.float32)
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
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
        if train:
            opt.zero_grad()
        with torch.set_grad_enabled(train):
            out  = model(x)
            loss = criterion(out, y)
        if train:
            loss.backward()
            opt.step()
        preds = torch.argmax(out, 1)
        total_correct += (preds == y).sum().item()
        total_loss    += loss.item() * x.size(0)
        total_n       += x.size(0)

    return total_loss / total_n, total_correct / total_n


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.append(torch.argmax(out, 1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def train_single_run(X_train, y_train, X_val, y_val, X_test, y_test,
                     batch_size, epochs, device, use_fft=False):
    train_ds = VibDataset(X_train, y_train, use_fft)
    val_ds   = VibDataset(X_val,   y_val,   use_fft)
    test_ds  = VibDataset(X_test,  y_test,  use_fft)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model     = CNN1D(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val   = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        tr_loss,  tr_acc  = run_epoch(model, train_loader, criterion, opt, device, True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion, opt, device, False)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)

    _, final_train = run_epoch(model, train_loader, criterion, None, device, False)
    _, final_val   = run_epoch(model, val_loader,   criterion, None, device, False)
    preds, true_labels = evaluate(model, test_loader, device)
    final_test = (preds == true_labels).mean()

    return final_train, final_val, final_test, preds, true_labels


def save_confusion_matrix_figure(y_true, y_pred, class_names, out_basepath,
                                  normalize=True, show_counts=True, title=None, dpi_png=600):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_plot = cm.astype(float)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums

    fig_w = max(8.5, 0.95 * len(class_names) + 3.0)
    fig_h = max(7.5, 0.85 * len(class_names) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im   = ax.imshow(cm_plot, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.tick_params(axis="both", which="major", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    if title is None:
        title = "Confusion Matrix" + (" (Normalized)" if normalize else "")
    ax.set_title(title, fontsize=13, pad=10)

    thresh = cm_plot.max() * 0.65
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                pct  = cm_plot[i, j] * 100.0
                text = f"{pct:.1f}%\n(n={cm[i, j]})" if show_counts else f"{pct:.1f}%"
            else:
                text = f"{cm[i, j]}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if cm_plot[i, j] > thresh else "black")

    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    ax.grid(False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_basepath) or ".", exist_ok=True)
    fig.savefig(out_basepath + ".pdf", bbox_inches="tight")
    fig.savefig(out_basepath + ".png", dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--channel",  type=str, default="X")
    parser.add_argument("--use_fft",  action="store_true")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--seeds",    type=int, default=5)
    parser.add_argument("--outdir",   type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Channel: {args.channel} | FFT: {args.use_fft} | Epochs: {args.epochs} | Seeds: {args.seeds}")
    print(f"Sweep: {len(BATCH_SIZES)} batch sizes x {len(WINDOW_SIZES)} window sizes "
          f"= {len(BATCH_SIZES)*len(WINDOW_SIZES)} combinations\n")

    print("Loading signals from .mat files...")
    signals = load_all_signals(args.data_dir, args.channel)
    print(f"Loaded {len(signals)} files.\n")

    os.makedirs(args.outdir, exist_ok=True)
    results = []

    best_test_mean = -1.0
    best_config    = None
    best_preds     = None
    best_labels    = None

    total_combos = len(BATCH_SIZES) * len(WINDOW_SIZES)
    combo_idx    = 0

    for ws in WINDOW_SIZES:
        samples, labels = create_windows(signals, ws)
        n_per_class = {}
        for lbl in np.unique(labels):
            n_per_class[CLASS_NAMES[lbl]] = (labels == lbl).sum()

        print(f"{'='*70}")
        print(f"Window Size: {ws} | Total windows: {len(samples)}")
        for name, cnt in n_per_class.items():
            print(f"  {name}: {cnt}")
        print(f"{'='*70}")

        for bs in BATCH_SIZES:
            combo_idx += 1
            print(f"\n[{combo_idx}/{total_combos}] Batch={bs}, Window={ws}")

            seed_train, seed_val, seed_test = [], [], []

            for s_idx in range(args.seeds):
                seed = 42 + s_idx
                set_seed(seed)

                X_train, X_temp, y_train, y_temp = train_test_split(
                    samples, labels, test_size=0.3, random_state=seed, stratify=labels
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
                )

                t0 = time.time()
                tr_acc, val_acc, test_acc, preds, true_labels = train_single_run(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    bs, args.epochs, device, args.use_fft,
                )
                elapsed = time.time() - t0

                seed_train.append(tr_acc)
                seed_val.append(val_acc)
                seed_test.append(test_acc)

                print(f"  Seed {seed}: Train {tr_acc*100:6.2f}% | Val {val_acc*100:6.2f}% | "
                      f"Test {test_acc*100:6.2f}% | {elapsed:.1f}s")

                if test_acc > best_test_mean:
                    best_test_mean = test_acc
                    best_config    = (bs, ws)
                    best_preds     = preds
                    best_labels    = true_labels

            tr_mean   = np.mean(seed_train) * 100
            val_mean  = np.mean(seed_val)   * 100
            test_mean = np.mean(seed_test)  * 100
            test_std  = np.std(seed_test)   * 100

            results.append({
                "Batch Size":             bs,
                "Input Signal Length":    ws,
                "Train Accuracy (%)":     f"{tr_mean:.2f}",
                "Validation Accuracy (%)": f"{val_mean:.2f}",
                "Test Accuracy (%) and Standard Deviation": f"{test_mean:.2f}±{test_std:.2f}",
            })

            print(f"  >> Mean: Train {tr_mean:.2f}% | Val {val_mean:.2f}% | "
                  f"Test {test_mean:.2f}±{test_std:.2f}%")

    import pandas as pd
    df   = pd.DataFrame(results)
    mode = "fft" if args.use_fft else "time"
    ch   = args.channel.upper()

    csv_path = os.path.join(args.outdir, f"sweep_results_{ch}_{mode}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults table saved: {csv_path}")

    xlsx_path = os.path.join(args.outdir, f"sweep_results_{ch}_{mode}.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"Results table saved: {xlsx_path}")

    print(f"\n{'='*70}")
    print(f"SWEEP RESULTS - Channel: {ch}, Mode: {mode}")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    bs_best, ws_best = best_config
    cm_path = os.path.join(args.outdir,
                           f"confusion_matrix_{ch}_{mode}_bs{bs_best}_ws{ws_best}")
    save_confusion_matrix_figure(
        y_true=best_labels, y_pred=best_preds, class_names=CLASS_NAMES,
        out_basepath=cm_path, normalize=True, show_counts=True,
        title=f"Confusion Matrix ({ch}, {mode}, batch={bs_best}, window={ws_best})",
    )
    print(f"\nBest config: Batch={bs_best}, Window={ws_best}")
    print(f"Confusion matrix saved: {cm_path}.pdf / .png")

    print(f"\nClassification Report (Best Config):")
    print(classification_report(best_labels, best_preds, target_names=CLASS_NAMES, zero_division=0))

    print("\nDone!")


if __name__ == "__main__":
    main()
