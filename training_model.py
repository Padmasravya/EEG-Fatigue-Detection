import sys, os, json
import numpy as np
import scipy.io
from scipy.signal import welch, butter, filtfilt, iirnotch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
MAT_PATH    = r"C:\\Users\\padma\\Downloads\\Biosignals_project\\EEG_drowsiness.mat"
OUTPUT_DIR  = r"C:\\Users\\padma\\Downloads\\Biosignals_project"

FS          = 128
N_SAMPLES   = 384       # samples per trial (3 seconds)

# Channels selected from discrimination analysis
CHANNELS    = {
    'FC5': 7,
    'C3' : 12,
    'CP5': 17,
}

CH_NAMES    = list(CHANNELS.keys())
CH_IDX      = list(CHANNELS.values())

BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 13.0),
    'beta' : (13.0,30.0),
    'gamma': (30.0,45.0),
}

# ============================================================
# FILTERS
# ============================================================
def remove_dc(sig):
    """
    Filter 1 — DC removal.
    Subtracts the mean to eliminate DC offset (electrode drift).
    Essential for EEG — DC shifts contaminate all frequency analysis.
    """
    return sig - np.mean(sig)

def bandpass_filter(sig, fs=FS, low=0.5, high=45.0, order=4):
    """
    Filter 2 — Butterworth bandpass 0.5–45 Hz.
    Removes:
      - Sub-0.5Hz slow drifts (sweat, movement artefact)
      - Above-45Hz high frequency noise and aliasing
    4th order gives -80dB/decade rolloff — standard for EEG.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)   # zero-phase (filtfilt) — no phase distortion

def notch_filter(sig, fs=FS, freq=50.0, Q=30.0):
    """
    Filter 3 — IIR notch at 50 Hz (Indian power line frequency).
    Removes power line interference which contaminates EEG.
    Q=30 gives narrow notch — removes 50Hz without affecting nearby bands.
    """
    nyq  = fs / 2.0
    w0   = freq / nyq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, sig)

def preprocess(sig, fs=FS):
    """Full preprocessing chain — applied to every EEG trial."""
    sig = remove_dc(sig)          # step 1: DC removal
    sig = bandpass_filter(sig, fs) # step 2: bandpass 0.5–45Hz
    sig = notch_filter(sig, fs)    # step 3: 50Hz notch
    return sig

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def bandpower(sig, fs, band):
    """
    Power Spectral Density via Welch's method.
    nperseg = fs*2 (256 points) gives 0.5Hz frequency resolution.
    Integrated over band using the trapezoidal rule.
    """
    freqs, psd = welch(sig, fs=fs, nperseg=fs * 2, window='hann')
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[idx], freqs[idx])

def extract_features(sig, fs=FS):
    """
    Extract features from one preprocessed EEG channel signal.

    Returns 7 features per channel:
      5 band powers + fatigue index + alpha/theta ratio
    """
    powers = {}
    for band_name, band_range in BANDS.items():
        powers[band_name] = bandpower(sig, fs, band_range)

    # Fatigue index: theta/(alpha+beta) — rises during drowsiness
    fi = powers['theta'] / (powers['alpha'] + powers['beta'] + 1e-9)

    # Alpha/theta ratio — inversely tracks arousal
    at_ratio = powers['alpha'] / (powers['theta'] + 1e-9)

    return [
        powers['delta'],
        powers['theta'],
        powers['alpha'],
        powers['beta'],
        powers['gamma'],
        fi,
        at_ratio,
    ]

FEATURE_NAMES = []
for ch in CH_NAMES:
    for f in ['delta','theta','alpha','beta','gamma','fatigue_idx','alpha_theta']:
        FEATURE_NAMES.append(f'{ch}_{f}')

# ============================================================
# BUILD FEATURE MATRIX
# ============================================================
def build_features(eeg, y):
    """
    eeg : (2022, 30, 384)
    y   : (2022,)
    Returns X (2022, 21), y (2022,)
    """
    print("[1/5] Preprocessing + feature extraction...")
    X = np.zeros((len(eeg), len(CH_IDX) * 7))

    for trial in range(len(eeg)):
        if trial % 300 == 0:
            print(f"      {trial}/{len(eeg)}", end='\r')
        feat_row = []
        for ch in CH_IDX:
            sig = eeg[trial, ch, :].astype(float)
            sig = preprocess(sig)
            feat_row.extend(extract_features(sig))
        X[trial] = feat_row

    print(f"\n      Done. Shape: {X.shape}")
    return X

# ============================================================
# TRAIN
# ============================================================
def train(X, y):
    print("\n[2/5] Training classifiers (5-fold CV)...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # SVM — RBF kernel
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)
    svm_f1  = cross_val_score(svm, X_scaled, y, cv=cv, scoring='f1')
    svm_acc = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"      SVM (RBF)      F1={svm_f1.mean():.4f}±{svm_f1.std():.4f}"
          f"  Acc={svm_acc.mean():.4f}±{svm_acc.std():.4f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_f1  = cross_val_score(rf, X_scaled, y, cv=cv, scoring='f1')
    rf_acc = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"      Random Forest  F1={rf_f1.mean():.4f}±{rf_f1.std():.4f}"
          f"  Acc={rf_acc.mean():.4f}±{rf_acc.std():.4f}")

    # Final fit on all data
    svm.fit(X_scaled, y)
    rf.fit(X_scaled, y)

    # Full classification report on training data (for report)
    print("\n      SVM full classification report:")
    print(classification_report(y, svm.predict(X_scaled),
                                 target_names=['Alert','Drowsy']))

    return svm, rf, scaler, svm_f1.mean(), rf_f1.mean()

# ============================================================
# EXPORT WEIGHTS → firmware-ready JSON + C header
# ============================================================
def export_weights(X, y, svm, scaler, output_dir):
    print("\n[3/5] Exporting weights for ESP32 firmware...")

    alert_mask  = y == 0
    drowsy_mask = y == 1

    # Per-channel fatigue index thresholds (alert mean + 1.5*std)
    fi_stats = {}
    for i, ch in enumerate(CH_NAMES):
        fi_col = i * 7 + 5   # fatigue_idx is index 5 in each channel's 7 features
        a_mean = float(X[alert_mask,  fi_col].mean())
        a_std  = float(X[alert_mask,  fi_col].std())
        d_mean = float(X[drowsy_mask, fi_col].mean())
        threshold = a_mean + 1.5 * a_std
        fi_stats[ch] = {
            'alert_mean' : a_mean,
            'alert_std'  : a_std,
            'drowsy_mean': d_mean,
            'threshold'  : threshold,
            'change_pct' : ((d_mean - a_mean) / a_mean) * 100,
        }
        print(f"      {ch}  alert={a_mean:.5f}  drowsy={d_mean:.5f}"
              f"  threshold={threshold:.5f}"
              f"  change={fi_stats[ch]['change_pct']:+.1f}%")

    # SVM support vectors are complex to port to C
    # Instead we use the fatigue index weighted score approach
    # (linear combination of per-channel FI — portable to ESP32)
    # Weight = change% normalized so they sum to 1
    changes  = np.array([fi_stats[ch]['change_pct'] for ch in CH_NAMES])
    weights  = changes / changes.sum()

    profile = {
        'fs'           : FS,
        'channels'     : CH_NAMES,
        'channel_idx'  : CH_IDX,
        'feature_names': FEATURE_NAMES,
        'fi_stats'     : fi_stats,
        'weights'      : {ch: float(w) for ch, w in zip(CH_NAMES, weights)},
        'scaler_mean'  : scaler.mean_.tolist(),
        'scaler_std'   : scaler.scale_.tolist(),
        'global_threshold': 0.0,   # score > 0 = drowsy (after weighting)
    }

    # Save JSON
    json_path = os.path.join(output_dir, 'model_profile.json')
    with open(json_path, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"\n      Saved model_profile.json")

    # Save C header for firmware
    header = generate_c_header(profile, fi_stats, weights)
    h_path = os.path.join(output_dir, 'model_weights.h')
    with open(h_path, 'w') as f:
        f.write(header)
    print(f"      Saved model_weights.h  (paste into Arduino project)")

    return profile

def generate_c_header(profile, fi_stats, weights):
    lines = []
    lines.append('/*')
    lines.append(' * model_weights.h')
    lines.append(' * Auto-generated by train_model.py')
    lines.append(' * EEG Fatigue Detection — trained on Figshare Drowsiness Dataset')
    lines.append(' * Channels: FC5 (idx 7), C3 (idx 12), CP5 (idx 17)')
    lines.append(' *')
    lines.append(' * Filters applied during training:')
    lines.append(' *   1. DC removal (zero-mean)')
    lines.append(' *   2. Butterworth bandpass 0.5-45Hz, 4th order, zero-phase')
    lines.append(' *   3. IIR notch 50Hz, Q=30')
    lines.append(' *')
    lines.append(' * Feature: fatigue index = theta/(alpha+beta)')
    lines.append(' * Score  : weighted sum of per-channel fatigue indices')
    lines.append(' * Label  : score > DROWSY_THRESHOLD => drowsy')
    lines.append(' */')
    lines.append('')
    lines.append('#ifndef MODEL_WEIGHTS_H')
    lines.append('#define MODEL_WEIGHTS_H')
    lines.append('')
    lines.append(f'#define FS            {FS}')
    lines.append(f'#define N_CHANNELS    {len(CH_NAMES)}')
    lines.append('')

    for ch in CH_NAMES:
        s = fi_stats[ch]
        lines.append(f'// {ch} — fatigue index stats')
        lines.append(f'#define {ch}_FI_ALERT_MEAN   {s["alert_mean"]:.6f}f')
        lines.append(f'#define {ch}_FI_ALERT_STD    {s["alert_std"]:.6f}f')
        lines.append(f'#define {ch}_FI_DROWSY_MEAN  {s["drowsy_mean"]:.6f}f')
        lines.append(f'#define {ch}_FI_THRESHOLD    {s["threshold"]:.6f}f')
        lines.append('')

    lines.append('// Channel weights (sum=1.0, proportional to drowsy change%)')
    for ch, w in zip(CH_NAMES, weights):
        lines.append(f'#define W_{ch:<4}  {w:.6f}f')
    lines.append('')

    lines.append('// Global score threshold — score above this = drowsy')
    lines.append('#define DROWSY_THRESHOLD  0.45f')
    lines.append('')

    # Normalization stats (per-channel FI mean/std from alert baseline)
    lines.append('// Normalization: (fi - alert_mean) / alert_std')
    lines.append('const float FI_MEAN[N_CHANNELS] = {')
    vals = [f'  {fi_stats[ch]["alert_mean"]:.6f}f' for ch in CH_NAMES]
    lines.append(',\n'.join(vals))
    lines.append('};')
    lines.append('')
    lines.append('const float FI_STD[N_CHANNELS] = {')
    vals = [f'  {fi_stats[ch]["alert_std"]:.6f}f' for ch in CH_NAMES]
    lines.append(',\n'.join(vals))
    lines.append('};')
    lines.append('')
    lines.append('const float CH_WEIGHTS[N_CHANNELS] = {')
    vals = [f'  {w:.6f}f' for w in weights]
    lines.append(',\n'.join(vals))
    lines.append('};')
    lines.append('')
    lines.append('#endif // MODEL_WEIGHTS_H')

    return '\n'.join(lines)

# ============================================================
# VISUALISATION
# ============================================================
def plot_results(eeg, y, X, profile):
    print("\n[4/5] Generating analysis plots...")
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    AC = '#4fc3f7'   # alert color
    DC = '#ef5350'   # drowsy color
    GC = '#2a2a2a'   # grid color
    TC = '#e0e0e0'   # text color

    alert_mask  = y == 0
    drowsy_mask = y == 1

    # Plot 1: Raw vs filtered signal
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#1a1a1a')
    trial_idx = np.where(alert_mask)[0][0]
    raw = eeg[trial_idx, CH_IDX[0], :].astype(float)
    t   = np.arange(N_SAMPLES) / FS
    ax1.plot(t, raw - raw.mean(),          color='#888', linewidth=0.8,
             alpha=0.6, label='Raw (DC removed)')
    ax1.plot(t, preprocess(raw),           color=AC,     linewidth=1.2,
             label='After bandpass + notch')
    ax1.set_title(f'FC5 — raw vs filtered (alert trial)',
                  color=TC, fontsize=11)
    ax1.set_xlabel('Time (s)', color=TC)
    ax1.set_ylabel('Amplitude', color=TC)
    ax1.legend(facecolor='#1a1a1a', labelcolor=TC, fontsize=9)
    ax1.tick_params(colors=TC)
    ax1.grid(color=GC, linewidth=0.5)
    for sp in ax1.spines.values(): sp.set_color(GC)

    # Plot 2: Power spectrum alert vs drowsy
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#1a1a1a')
    alert_trials  = np.where(alert_mask)[0][:50]
    drowsy_trials = np.where(drowsy_mask)[0][:50]
    pa_avg = np.zeros(N_SAMPLES + 1)
    pd_avg = np.zeros(N_SAMPLES + 1)
    for ti in alert_trials:
        sig = preprocess(eeg[ti, CH_IDX[0], :].astype(float))
        f, p = welch(sig, fs=FS, nperseg=FS*2)
        pa_avg[:len(p)] += p
    for ti in drowsy_trials:
        sig = preprocess(eeg[ti, CH_IDX[0], :].astype(float))
        f, p = welch(sig, fs=FS, nperseg=FS*2)
        pd_avg[:len(p)] += p
    pa_avg /= len(alert_trials)
    pd_avg /= len(drowsy_trials)
    mask = f <= 45
    ax2.semilogy(f[mask], pa_avg[:mask.sum()], color=AC,  linewidth=1.5, label='Alert')
    ax2.semilogy(f[mask], pd_avg[:mask.sum()], color=DC,  linewidth=1.5, label='Drowsy')
    for bn, (lo, hi) in BANDS.items():
        ax2.axvspan(lo, hi, alpha=0.06, color='white')
        ax2.text((lo+hi)/2, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 1e-3,
                 bn[:2], color='#666', fontsize=7, ha='center')
    ax2.set_title('Power spectrum FC5 (avg 50 trials)', color=TC, fontsize=11)
    ax2.set_xlabel('Frequency (Hz)', color=TC)
    ax2.legend(facecolor='#1a1a1a', labelcolor=TC, fontsize=9)
    ax2.tick_params(colors=TC)
    ax2.grid(color=GC, linewidth=0.5)
    for sp in ax2.spines.values(): sp.set_color(GC)

    # Plot 3–5: Fatigue index distribution per channel
    for i, (ch, ch_i) in enumerate(CHANNELS.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor('#1a1a1a')
        fi_col = i * 7 + 5
        ax.hist(X[alert_mask,  fi_col], bins=40, alpha=0.75,
                color=AC, label='Alert',  density=True)
        ax.hist(X[drowsy_mask, fi_col], bins=40, alpha=0.75,
                color=DC, label='Drowsy', density=True)
        thresh = profile['fi_stats'][ch]['threshold']
        ax.axvline(thresh, color='#ffd54f', linewidth=1.5,
                   linestyle='--', label=f'Threshold={thresh:.3f}')
        chg = profile['fi_stats'][ch]['change_pct']
        ax.set_title(f'{ch} fatigue index  ({chg:+.1f}% drowsy)',
                     color=TC, fontsize=10)
        ax.legend(facecolor='#1a1a1a', labelcolor=TC, fontsize=8)
        ax.tick_params(colors=TC)
        ax.grid(color=GC, linewidth=0.5)
        for sp in ax.spines.values(): sp.set_color(GC)

    fig.suptitle('EEG Fatigue Detection — Training Analysis\n'
                 'Dataset: Figshare Driver Drowsiness (11 subjects, driving task)',
                 color=TC, fontsize=13, fontweight='bold')

    out = os.path.join(OUTPUT_DIR, 'training_analysis.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"      Saved training_analysis.png")
    plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("EEG FATIGUE DETECTION — TRAINING PIPELINE")
    print("=" * 55)
    print()

    # Load
    print("Loading dataset...")
    mat = scipy.io.loadmat(MAT_PATH)
    eeg = mat['EEGsample']          # (2022, 30, 384)
    y   = mat['substate'].ravel()   # (2022,) 0=alert 1=drowsy
    print(f"  Trials: {len(eeg)}  "
          f"Alert: {(y==0).sum()}  Drowsy: {(y==1).sum()}")
    print()

    # Build features
    X = build_features(eeg, y)

    # Train
    svm, rf, scaler, svm_f1, rf_f1 = train(X, y)

    # Export
    profile = export_weights(X, y, svm, scaler, OUTPUT_DIR)

    # Plot
    plot_results(eeg, y, X, profile)

    # Summary
    print("\n[5/5] Summary")
    print("=" * 55)
    print(f"  Best model : {'SVM' if svm_f1 >= rf_f1 else 'Random Forest'}")
    print(f"  SVM F1     : {svm_f1:.4f}")
    print(f"  RF  F1     : {rf_f1:.4f}")
    print()
    print("  Files saved:")
    print(f"    model_profile.json  — thresholds + weights")
    print(f"    model_weights.h     — paste into Arduino firmware")
    print(f"    training_analysis.png")
    print()
    print("  Next step: open model_weights.h and paste into")
    print("  your Arduino project folder, then run firmware.ino")

if __name__ == "__main__":
    main()
