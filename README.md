# UAV Rotor Blade Fault Diagnosis with 1D CNN

This repository contains the Python code developed for

*Deep Learning Fault Diagnosis of Multirotor UAV Rotor Blades Using Triaxial Accelerometer Data*

---

## Overview

This project implements a lightweight **1D Convolutional Neural Network (1D CNN)** for vibration-based rotor blade fault diagnosis in multirotor UAVs. The model works directly on raw time-domain accelerometer signals, without any manual feature extraction.

The model was evaluated on two different datasets to test how well the approach generalizes across different experimental conditions:

- **Dataset 1 :** 5-class problem using triaxial accelerometer recordings from a DJI Mini 2 in indoor hover flight. Best result: **98.58% test accuracy** (batch size 128, window length 4096).
- **Dataset 2 :** 10-class problem from ground testing with three fault types at three severity levels. Best result: **68.67% test accuracy** (batch size 64, window length 2048).

---

## Key Findings

- Raw time-domain input consistently outperformed RMS and kurtosis feature representations by **25 to 41 percentage points** across both datasets.
- A critical failure mode was identified: when batch size approaches the total number of training samples, accuracy collapses by more than 20 percentage points. Batch size and window length must be selected together.
- The 1D CNN architecture is lightweight enough for potential embedded deployment on modern microcontrollers.

---

## Repository Structure

```
├── main_dataset1.py          # Training and evaluation for Dataset 1
├── ablation_dataset1.py      # Ablation study for Dataset 1 (raw vs. RMS vs. kurtosis)
├── main_dataset2.py          # Training and evaluation for Dataset 2
├── ablation_dataset2.py      # Ablation study for Dataset 2 (raw vs. RMS vs. kurtosis)
```

---

## Model Architecture

A 3-block 1D CNN with the following structure:

```
Input (1 x window_size)
  → Conv1D(16, k=7) → BatchNorm → ReLU → MaxPool
  → Conv1D(32, k=5) → BatchNorm → ReLU → MaxPool
  → Conv1D(64, k=3) → BatchNorm → ReLU → AdaptiveMaxPool
  → Linear(64) → ReLU → Dropout(0.5)
  → Linear(num_classes)
```

---

## Requirements

```
torch
numpy
pandas
scipy
scikit-learn
matplotlib
openpyxl
```

Install with:

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib openpyxl
```

---

## Usage

**Dataset 1 - Training:**
```bash
python main_dataset1.py --channel Z --batch 128 --epochs 50
```

**Dataset 1 - Ablation study (all methods and axes):**
```bash
python ablation.py --run_all --window 4096 --batch 128 --epochs 50
```

**Dataset 2 - Training sweep (all batch/window combinations):**
```bash
python main_dataset2.py --channel Y --epochs 50 --seeds 5
```

**Dataset 2 - Ablation study:**
```bash
python dronepropb_ablation.py --run_all --epochs 50
```

---

## Datasets

- **Dataset 1:** Multiaxial vibration data for blade fault diagnosis in multirotor UAVs — Al-Haddad et al. (2025), *Scientific Data*. [DOI: 10.1038/s41597-025-05692-4](https://doi.org/10.1038/s41597-025-05692-4)
- **Dataset 2:** TBA

---

## Author

**Irmak Tankaya**  

