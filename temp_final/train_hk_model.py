"""
Hollow Knight SSVEP — Model Training Script
=============================================
Loads calibration EEG data collected by hollow_knight_ssvep.py,
trains an FBTRCA classifier across all 8 targets, evaluates with
Leave-One-Out cross-validation, and saves the model to disk.

Usage:
    python train_hk_model.py

Dependencies:
    pip install numpy scipy mne scikit-learn pandas brainda tqdm
"""

import os, pickle
import numpy as np
import pandas as pd
import mne
from collections import OrderedDict
from sklearn.pipeline import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from brainda.algorithms.utils.model_selection import (
    set_random_seeds, generate_loo_indices, match_loo_indices)
from brainda.algorithms.decomposition import (
    FBTRCA, generate_filterbank)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG  — mirror values from hollow_knight_ssvep.py
# ─────────────────────────────────────────────
folder_path    = 'data/hollow_knight_ssvep/sub-01/ses-01/'
model_save_dir = 'cache/'
model_name     = 'FBTRCA_model_hk.pkl'

sampling_rate  = 250
stim_duration  = 1.2        # seconds
n_per_class    = 2          # repetitions per class used during calibration
N_CLASSES      = 8

# 8 stimulus classes matching the stimulus script
stimulus_classes = [
    (8,  0.0),
    (9,  0.0),
    (10, 0.0),
    (11, 0.0),
    (8,  0.5),
    (9,  0.5),
    (10, 0.5),
    (11, 0.5),
]

# Human-readable labels for plots
ACTION_LABELS = [
    'Left+Jump', 'Jump', 'Dash', 'Right+Jump',
    'Left',      'Attack', 'Focus', 'Right',
]

# ─────────────────────────────────────────────
# 1. DISCOVER & LOAD RUN FILES
# ─────────────────────────────────────────────
run_files = sorted([
    f for f in os.listdir(folder_path)
    if f.startswith(f'eeg-trials_{n_per_class}-per-class_run-') and f.endswith('.npy')
])

if not run_files:
    raise FileNotFoundError(
        f'No EEG trial files found in {folder_path}\n'
        f'Run hollow_knight_ssvep.py in calibration_mode=True first.'
    )

print(f'Found {len(run_files)} run file(s): {run_files}')

# ─────────────────────────────────────────────
# 2. REVERSE SHUFFLE & RESHAPE EACH RUN
# ─────────────────────────────────────────────
# During recording, trials were stored in shuffled order.
# We reverse the shuffle to restore (rep, class, channel, time) ordering.

n_timepoints = int(stim_duration * sampling_rate)   # 300
n_channels   = 8

reverted_list = []

for run_file in run_files:
    run_number  = int(run_file.split('-run-')[1].split('.')[0])
    trial_data  = np.load(os.path.join(folder_path, run_file), allow_pickle=True)

    # trial_data shape: (n_per_class * N_CLASSES, n_channels, n_timepoints)
    n_total = trial_data.shape[0]

    np.random.seed(run_number)
    shuffled_idx = np.random.permutation(n_total)

    reverted = np.empty_like(trial_data)
    reverted[shuffled_idx] = trial_data   # undo shuffle

    # Reshape to (n_per_class, N_CLASSES, n_channels, n_timepoints)
    reverted = reverted.reshape(n_per_class, N_CLASSES, n_channels, n_timepoints)
    reverted_list.append(reverted)
    print(f'  Run {run_number}: shape after revert = {reverted.shape}')

# Combined shape: (total_reps, N_CLASSES, n_channels, n_timepoints)
combined = np.concatenate(reverted_list, axis=0)
print(f'\nCombined EEG shape: {combined.shape}')
# e.g. (total_reps=2, 8, 8, 300)

# ─────────────────────────────────────────────
# 3. BASELINE CORRECTION
# ─────────────────────────────────────────────
# The trial data was already cropped to exclude the baseline period
# in the stimulus script (collect_trial_eeg returns cropped).
# No further baseline subtraction needed here.
# If you saved the full trial (including baseline), uncomment below:

# baseline_samples = int(0.2 * sampling_rate)
# baseline_avg = np.mean(combined[..., :baseline_samples], axis=-1, keepdims=True)
# combined = combined - baseline_avg
# combined = combined[..., baseline_samples:]

# ─────────────────────────────────────────────
# 4. BUILD METADATA & LABELS
# ─────────────────────────────────────────────
n_reps   = combined.shape[0]
target_tab = {tuple(map(float, cls)): idx for idx, cls in enumerate(stimulus_classes)}

# y: label for each (class × rep) epoch
y = np.array([list(target_tab.values())] * n_reps).T.reshape(-1)

# X: (n_reps*N_CLASSES, n_channels, n_timepoints)
X = combined.swapaxes(0, 1).reshape(-1, n_channels, n_timepoints)
X = X - np.mean(X, axis=-1, keepdims=True)   # zero-mean each epoch

print(f'X shape: {X.shape}  |  y shape: {y.shape}')

# ─────────────────────────────────────────────
# 5. FILTER BANK  (3 sub-bands)
# ─────────────────────────────────────────────
# Sub-band k covers [8k, 90] Hz, targeting fundamental + harmonics
n_bands = 3
wp = [[8 * i, 90]     for i in range(1, n_bands + 1)]
ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
filterbank    = generate_filterbank(wp, ws, sampling_rate, order=4, rp=1)
filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25
print(f'Filter bank: {len(filterbank)} bands  |  weights: {np.round(filterweights, 3)}')

# ─────────────────────────────────────────────
# 6. LEAVE-ONE-OUT CROSS-VALIDATION
# ─────────────────────────────────────────────
events = []
for j_class in range(N_CLASSES):
    events.extend([str(stimulus_classes[j_class]) for _ in range(n_reps)])
events = np.array(events)

subjects = ['1'] * (N_CLASSES * n_reps)
meta     = pd.DataFrame(
    data=np.array([subjects, events]).T,
    columns=['subject', 'event']
)

set_random_seeds(42)
loo_indices = generate_loo_indices(meta)
n_loo       = len(loo_indices['1'][events[0]])

fbtrca_model = FBTRCA(filterbank, filterweights=filterweights, ensemble=True)

loo_accs   = []
all_preds  = []
all_truths = []
final_model = None   # will hold the last fully-trained model

print(f'\nRunning {n_loo}-fold Leave-One-Out CV ...')

for k in range(n_loo):
    train_ind, val_ind, test_ind = match_loo_indices(k, meta, loo_indices)
    train_ind = np.concatenate([train_ind, val_ind])

    trainX, trainY = X[train_ind], y[train_ind]
    testX,  testY  = X[test_ind],  y[test_ind]

    m = clone(fbtrca_model).fit(trainX, trainY)
    preds = m.predict(testX)

    fold_acc = balanced_accuracy_score(testY, preds)
    loo_accs.append(fold_acc)
    all_preds.extend(preds)
    all_truths.extend(testY)

    print(f'  Fold {k+1}/{n_loo}  balanced_acc={fold_acc:.3f}')
    final_model = m   # keep the last fold's model

mean_acc = np.mean(loo_accs)
overall  = accuracy_score(all_truths, all_preds)
print(f'\nLOO Mean Balanced Accuracy : {mean_acc*100:.2f}%')
print(f'LOO Overall Accuracy       : {overall*100:.2f}%')

# ─────────────────────────────────────────────
# 7. TRAIN FINAL MODEL ON ALL DATA
# ─────────────────────────────────────────────
print('\nTraining final model on all data ...')
final_model = clone(fbtrca_model).fit(X, y)
print('Done.')

# ─────────────────────────────────────────────
# 8. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
cm = confusion_matrix(all_truths, all_preds, normalize='true')

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(ACTION_LABELS, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ACTION_LABELS, fontsize=9)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'FBTRCA LOO Confusion Matrix\n'
             f'Mean Balanced Acc: {mean_acc*100:.1f}%  |  Overall: {overall*100:.1f}%')

for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, f'{cm[i,j]:.2f}',
                ha='center', va='center',
                color='white' if cm[i, j] > 0.5 else 'black',
                fontsize=8)

plt.tight_layout()
cm_path = os.path.join(model_save_dir, 'confusion_matrix_hk.png')
os.makedirs(model_save_dir, exist_ok=True)
plt.savefig(cm_path, dpi=150)
print(f'Confusion matrix saved to {cm_path}')
plt.show()

# ─────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────
model_path = os.path.join(model_save_dir, model_name)
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f'Model saved to {model_path}')
print('\nNext step: set calibration_mode=False in hollow_knight_ssvep.py and play!')
