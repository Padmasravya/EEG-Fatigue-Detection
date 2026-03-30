"""
================================================
EEG FATIGUE DETECTION — PYTHON ONLY PIPELINE
================================================
Dataset : Figshare EEG Driver Drowsiness (14273687)
          2022 trials x 30 channels x 384 samples @ 128Hz
          substate: 0=alert, 1=drowsy
          11 subjects in contiguous blocks (verified)

Channels: FC5 (idx 7), C3 (idx 12), CP5 (idx 17)
          Selected by fatigue index discrimination:
          FC5=+38.0%  C3=+37.4%  CP5=+36.8%

Signal processing (mirrors training):
  1. DC removal      — subtract trial mean
  2. Bandpass filter — 0.5-45Hz Butterworth 4th order
                       zero-phase (filtfilt)
  3. Notch filter    — 50Hz IIR Q=30

Feature:
  Welch PSD (nperseg=256, 0.5Hz resolution, Hann window)
  Fatigue index = theta / (alpha + beta)
  theta: 4-8Hz  alpha: 8-13Hz  beta: 13-30Hz

Robust adaptive calibration:
  First 40 alert-phase trials -> personal median + IQR
  No alerts during warmup (5) or calibration (40) phase

Alert thresholds (robust z-score):
  WATCH > 1.0  WARN > 2.0  HIGH > 3.0
  OR N_CONFIRM=4 consecutive WATCH+

Output:
  - Live dashboard (updates every trial)
  - Session CSV log
  - Telegram alert on WARN/HIGH
  - Full analysis report (saved at end):
      confusion matrix, ROC curve, score distributions,
      fatigue index timeline, band power comparison,
      per-subject accuracy breakdown
================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import scipy.io
from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy import stats
from sklearn.metrics import (confusion_matrix, roc_curve,
                              auc, classification_report,
                              ConfusionMatrixDisplay)
import requests
import csv
import time
from datetime import datetime
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
MAT_PATH    = r"C:\\Users\\padma\\Downloads\\Biosignals_project\\EEG_drowsiness.mat"
BOT_TOKEN   = "7879868798:AAGQtL_DH5mPV6NaDkl_wA3Q23OXiZ2-fAw"
CHAT_ID     = "6347671410"
SAVE_DIR    = r"C:\\Users\\padma\\Downloads\\Biosignals_project"

FS          = 128
N_SAMPLES   = 384

CH_NAMES    = ['FC5', 'C3', 'CP5']
CH_IDX      = [7, 12, 17]

BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 13.0),
    'beta' : (13.0,30.0),
    'gamma': (30.0,45.0),
}

# Subject block boundaries (verified from dataset)
SUBJECT_BLOCKS = {
    1:  (0,    187),
    2:  (188,  319),
    3:  (320,  469),
    4:  (470,  617),
    5:  (618,  841),
    6:  (842,  1007),
    7:  (1008, 1109),
    8:  (1110, 1373),
    9:  (1374, 1687),
    10: (1688, 1795),
    11: (1796, 2021),
}

# Trained global constants (from model_profile.json)
TRAINED_FI_MEANS = np.array([0.364776, 0.373102, 0.336537])
TRAINED_FI_STDS  = np.array([0.317643, 0.266618, 0.192442])
CH_WEIGHTS       = np.array([0.338509, 0.333602, 0.327889])

# Calibration
WARMUP_TRIALS      = 5
CALIBRATION_TRIALS = 40

# Alert thresholds
SCORE_WATCH  = 0.2
SCORE_WARN   = 0.7
SCORE_HIGH   = 1.3
N_CONFIRM    = 2
COOLDOWN_SEC = 10

EMA_ALPHA = 0.5

# ============================================================
# FILTERS
# ============================================================
def remove_dc(sig):
    """DC removal — subtracts mean. Removes electrode drift."""
    return sig - sig.mean()

def bandpass_filter(sig, fs=FS, low=0.5, high=45.0, order=4):
    """
    Butterworth bandpass 0.5-45Hz, 4th order, zero-phase.
    Removes sub-0.5Hz motion artefacts and above-45Hz noise.
    """
    nyq  = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def notch_filter(sig, fs=FS, freq=50.0, Q=30.0):
    """
    IIR notch 50Hz, Q=30.
    Removes Indian power line interference.
    """
    nyq  = fs / 2.0
    b, a = iirnotch(freq/nyq, Q)
    return filtfilt(b, a, sig)

def preprocess(sig):
    """DC removal -> bandpass 0.5-45Hz -> notch 50Hz."""
    return notch_filter(bandpass_filter(remove_dc(sig)))

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def bandpower(sig, fs, band):
    """Welch PSD integrated over band. nperseg=256 -> 0.5Hz res."""
    freqs, psd = welch(sig, fs=fs, nperseg=fs*2, window='hann')
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[idx], freqs[idx])

def compute_band_powers(sig):
    return {name: bandpower(sig, FS, rng)
            for name, rng in BANDS.items()}

def fatigue_index(powers):
    """
    theta / (alpha + beta)
    Theta rises during drowsiness. Beta falls with fatigue.
    Ratio amplifies both — robust to amplitude artefacts.
    """
    return powers['theta'] / (powers['alpha'] + powers['beta'] + 1e-9)

def theta_alpha_ratio(powers):
    return powers['theta'] / (powers['alpha'] + 1e-9)

def compute_score(fi_per_ch, medians, iqr_stds):
    """Robust z-score weighted sum across channels."""
    fi_vals = np.array([fi_per_ch[ch] for ch in CH_NAMES])
    fi_norm = (fi_vals - medians) / (iqr_stds + 1e-9)
    return float(np.dot(CH_WEIGHTS, fi_norm))

# ============================================================
# ROBUST CALIBRATION
# ============================================================
class Calibrator:
    """
    Personal baseline from first CALIBRATION_TRIALS alert trials.
    Uses median + IQR/1.349 (robust to outlier trials).
    Dataset is subject-blocked so all calibration trials
    come from Subject 1.
    """
    def __init__(self):
        self.fi_buffer   = {ch: [] for ch in CH_NAMES}
        self.done        = False
        self.medians     = TRAINED_FI_MEANS.copy()
        self.iqr_stds    = TRAINED_FI_STDS.copy()
        self.n_collected = 0

    def update(self, fi_per_ch, trial_idx):
        if self.done or trial_idx < WARMUP_TRIALS:
            return
        for ch in CH_NAMES:
            self.fi_buffer[ch].append(fi_per_ch[ch])
        self.n_collected += 1
        if self.n_collected >= CALIBRATION_TRIALS:
            self._finalize()

    def _finalize(self):
        new_med = np.zeros(len(CH_NAMES))
        new_iqr = np.zeros(len(CH_NAMES))
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE — Robust personal baseline")
        print("="*60)
        print(f"  {'CH':<5} {'Median':>10} {'IQR/1.349':>10} "
              f"{'Trained':>10} {'Shift':>8}")
        print("-"*60)
        for i, ch in enumerate(CH_NAMES):
            vals       = np.array(self.fi_buffer[ch])
            med        = float(np.median(vals))
            q75, q25   = np.percentile(vals, [75, 25])
            iqr_std    = max((q75-q25)/1.349, 0.05)
            new_med[i] = med
            new_iqr[i] = iqr_std
            shift = ((med - TRAINED_FI_MEANS[i])
                     / TRAINED_FI_MEANS[i]) * 100
            print(f"  {ch:<5} {med:>10.5f} {iqr_std:>10.5f} "
                  f"{TRAINED_FI_MEANS[i]:>10.5f} {shift:>+7.1f}%")
        self.medians  = new_med
        self.iqr_stds = new_iqr
        self.done     = True
        print("="*60 + "\n")

    @property
    def ready(self): return self.done

    def status(self, trial_idx):
        if trial_idx < WARMUP_TRIALS: return 'warmup'
        if not self.done:
            return f'calibrating ({CALIBRATION_TRIALS-self.n_collected} left)'
        return 'calibrated'

# ============================================================
# DECISION
# ============================================================
def get_level(score, drowsy_count, calib_ready, trial_idx):
    if trial_idx < WARMUP_TRIALS or not calib_ready:
        return 'OK'
    if score > SCORE_HIGH:   return 'HIGH'
    if drowsy_count >= N_CONFIRM: return 'HIGH'
    if score > SCORE_WARN:   return 'WARN'
    if score > SCORE_WATCH:  return 'WATCH'
    return 'OK'

def get_subject(trial_idx):
    for subj, (start, end) in SUBJECT_BLOCKS.items():
        if start <= trial_idx <= end:
            return subj
    return -1

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id": CHAT_ID,
                                  "text": msg}, timeout=5)
        print("[TELEGRAM] Sent.")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

# ============================================================
# SESSION LOGGER
# ============================================================
class SessionLogger:
    def __init__(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = f"{SAVE_DIR}\\session_{ts}.csv"
        self.f = open(self.path, 'w', newline='',
                      encoding='utf-8')
        self.w = csv.writer(self.f)
        self.w.writerow([
            'timestamp', 'trial', 'subject',
            'python_score', 'FC5_fi', 'C3_fi', 'CP5_fi',
            'theta_FC5', 'alpha_FC5', 'beta_FC5',
            'level', 'true_label', 'calib_status'
        ])
        print(f"[LOG] {self.path}")

    def log(self, trial, subject, score, fi_per_ch,
            bp_fc5, level, true_label, calib_status):
        self.w.writerow([
            datetime.now().isoformat(), trial, subject,
            f"{score:.4f}",
            f"{fi_per_ch['FC5']:.4f}",
            f"{fi_per_ch['C3']:.4f}",
            f"{fi_per_ch['CP5']:.4f}",
            f"{bp_fc5['theta']:.4f}",
            f"{bp_fc5['alpha']:.4f}",
            f"{bp_fc5['beta']:.4f}",
            level, true_label, calib_status
        ])
        self.f.flush()

    def close(self):
        self.f.close()
        print(f"[LOG] Saved: {self.path}")

# ============================================================
# LIVE DASHBOARD
# ============================================================
class Dashboard:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#0f0f0f')
        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               hspace=0.5, wspace=0.35)
        self.ax_eeg   = self.fig.add_subplot(gs[0, :])
        self.ax_bands = self.fig.add_subplot(gs[1, :2])
        self.ax_fi    = self.fig.add_subplot(gs[1, 2])
        self.ax_score = self.fig.add_subplot(gs[2, :2])
        self.ax_state = self.fig.add_subplot(gs[2, 2])

        TC = '#e0e0e0'; GC = '#2a2a2a'
        for ax in [self.ax_eeg, self.ax_bands,
                   self.ax_fi, self.ax_score, self.ax_state]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors=TC)
            for sp in ax.spines.values(): sp.set_color(GC)

        self.fi_history    = deque(maxlen=100)
        self.score_history = deque(maxlen=100)
        self.label_history = deque(maxlen=100)
        plt.ion(); plt.show()

    def update(self, trial_data, bpowers, fi_per_ch,
               score, level, true_label, calib_status,
               calib, subject):
        AC='#4fc3f7'; DC='#ef5350'
        GC='#2a2a2a'; TC='#e0e0e0'
        CH_COLS=['#ffd54f','#4fc3f7','#ef9a9a']
        t = np.arange(N_SAMPLES) / FS

        # EEG signal
        self.ax_eeg.cla(); self.ax_eeg.set_facecolor('#1a1a1a')
        for i, ch in enumerate(CH_NAMES):
            col = AC if true_label == 0 else DC
            self.ax_eeg.plot(t, trial_data[i] + i*40,
                             color=col, lw=0.8, alpha=0.9,
                             label=ch)
        self.ax_eeg.set_title(
            f'EEG — Subject {subject} | '
            f'True: {"ALERT" if true_label==0 else "DROWSY"} | '
            f'Score={score:+.3f} | {calib_status}',
            color=TC, fontsize=9)
        self.ax_eeg.legend(facecolor='#1a1a1a',
                           labelcolor=TC, fontsize=8,
                           loc='upper right')
        self.ax_eeg.set_xlabel('Time (s)', color=TC)
        self.ax_eeg.tick_params(colors=TC)
        for sp in self.ax_eeg.spines.values(): sp.set_color(GC)

        # Band powers
        self.ax_bands.cla()
        self.ax_bands.set_facecolor('#1a1a1a')
        bnames = list(BANDS.keys())
        x = np.arange(len(bnames)); w = 0.25
        for i, ch in enumerate(CH_NAMES):
            vals = [bpowers[ch][b] for b in bnames]
            self.ax_bands.bar(x+i*w, vals, w,
                              label=ch, color=CH_COLS[i],
                              alpha=0.85)
        self.ax_bands.set_xticks(x+w)
        self.ax_bands.set_xticklabels(bnames, color=TC)
        self.ax_bands.set_title('Band powers — Welch PSD',
                                color=TC, fontsize=10)
        self.ax_bands.legend(facecolor='#1a1a1a',
                             labelcolor=TC, fontsize=8)
        self.ax_bands.tick_params(colors=TC)
        for sp in self.ax_bands.spines.values(): sp.set_color(GC)

        # FI history
        self.fi_history.append(
            {ch: fi_per_ch[ch] for ch in CH_NAMES})
        self.ax_fi.cla(); self.ax_fi.set_facecolor('#1a1a1a')
        for i, ch in enumerate(CH_NAMES):
            self.ax_fi.plot(
                [d[ch] for d in self.fi_history],
                color=CH_COLS[i], lw=1.2, label=ch)
        if calib.ready:
            for i, ch in enumerate(CH_NAMES):
                self.ax_fi.axhline(
                    calib.medians[i] + calib.iqr_stds[i],
                    color=CH_COLS[i], lw=0.8,
                    ls='--', alpha=0.6)
        self.ax_fi.set_title('Fatigue index',
                             color=TC, fontsize=10)
        self.ax_fi.legend(facecolor='#1a1a1a',
                          labelcolor=TC, fontsize=7)
        self.ax_fi.tick_params(colors=TC)
        for sp in self.ax_fi.spines.values(): sp.set_color(GC)

        # Score timeline
        self.score_history.append(score)
        self.label_history.append(true_label)
        self.ax_score.cla()
        self.ax_score.set_facecolor('#1a1a1a')
        sc = list(self.score_history)
        lb = list(self.label_history)
        for i in range(len(sc)-1):
            self.ax_score.plot(
                [i, i+1], [sc[i], sc[i+1]],
                color=DC if lb[i]==1 else AC, lw=1.2)
        for thresh, col, lbl in [
            (SCORE_WATCH,'#ffd54f',f'WATCH({SCORE_WATCH})'),
            (SCORE_WARN, '#ff8a65',f'WARN({SCORE_WARN})'),
            (SCORE_HIGH, '#ef5350',f'HIGH({SCORE_HIGH})')]:
            self.ax_score.axhline(thresh, color=col,
                lw=0.8, ls='--', alpha=0.7, label=lbl)
        self.ax_score.set_title(
            'Fatigue score (blue=alert, red=drowsy)',
            color=TC, fontsize=9)
        self.ax_score.set_ylabel('Robust z-score', color=TC)
        self.ax_score.legend(facecolor='#1a1a1a',
                             labelcolor=TC, fontsize=7)
        self.ax_score.tick_params(colors=TC)
        for sp in self.ax_score.spines.values(): sp.set_color(GC)

        # Alert indicator
        self.ax_state.cla()
        self.ax_state.set_facecolor('#1a1a1a')
        self.ax_state.set_xlim(0,1); self.ax_state.set_ylim(0,1)
        self.ax_state.axis('off')
        col = {'OK':'#4fc3f7','WATCH':'#ffd54f',
                'WARN':'#ff8a65','HIGH':'#ef5350'}.get(level,'#555')
        self.ax_state.add_patch(
            plt.Circle((0.5,0.55),0.32,color=col,alpha=0.9))
        self.ax_state.text(0.5,0.55,level,ha='center',
            va='center',fontsize=16,fontweight='bold',
            color='white')
        self.ax_state.text(0.5,0.12,f'Subj {subject}',
            ha='center',va='center',fontsize=8,color='#aaa')
        self.ax_state.set_title('Alert level',
                                color=TC,fontsize=10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ============================================================
# ANALYSIS REPORT — generated after streaming ends
# ============================================================
def generate_report(records, save_dir):
    """
    Professional analysis plots saved as PNG files.
    All plots use real logged data — nothing fabricated.

    Plots generated:
      1. Confusion matrix
      2. ROC curve with AUC
      3. Score distribution (alert vs drowsy)
      4. Fatigue index timeline with ground truth
      5. Band power comparison (alert vs drowsy)
      6. Per-subject accuracy breakdown
    """
    import os
    print("\nGenerating analysis report...")

    # Only use calibrated trials for evaluation
    calib_records = [r for r in records if r['calib_status']=='calibrated']
    if len(calib_records) < 10:
        print("Not enough calibrated trials for report.")
        return

    scores     = np.array([r['score']     for r in calib_records])
    true_labels= np.array([r['true_label']for r in calib_records])
    fi_fc5     = np.array([r['fi_FC5']    for r in calib_records])
    fi_c3      = np.array([r['fi_C3']     for r in calib_records])
    fi_cp5     = np.array([r['fi_CP5']    for r in calib_records])
    theta_fc5  = np.array([r['theta_FC5'] for r in calib_records])
    alpha_fc5  = np.array([r['alpha_FC5'] for r in calib_records])
    beta_fc5   = np.array([r['beta_FC5']  for r in calib_records])
    subjects   = np.array([r['subject']   for r in calib_records])
    levels     = np.array([r['level']     for r in calib_records])

    # Binary predictions from score threshold
    # Use SCORE_WATCH as decision boundary
    pred_labels = (scores > SCORE_WATCH).astype(int)

    BG  = '#0f0f0f'; AC='#4fc3f7'; DC='#ef5350'
    TC  = '#e0e0e0'; GC = '#2a2a2a'

    # ── 1. CONFUSION MATRIX ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7,6),
                           facecolor=BG)
    ax.set_facecolor('#1a1a1a')
    cm  = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Alert','Drowsy'])
    disp.plot(ax=ax, colorbar=False,
              cmap='Blues')
    ax.set_title('Confusion Matrix\n'
                 '(calibrated trials only)',
                 color=TC, fontsize=12, pad=12)
    ax.xaxis.label.set_color(TC)
    ax.yaxis.label.set_color(TC)
    ax.tick_params(colors=TC)
    ax.title.set_color(TC)
    for text in disp.text_.ravel():
        text.set_color('white')
        text.set_fontsize(14)
    plt.tight_layout()
    p1 = os.path.join(save_dir, 'report_1_confusion_matrix.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.close()
    print(f"  Saved: report_1_confusion_matrix.png")

    # ── 2. ROC CURVE ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7,6), facecolor=BG)
    ax.set_facecolor('#1a1a1a')
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='#ffd54f', lw=2,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1], color='#555', lw=1,
            ls='--', label='Random classifier')
    # Mark operating point at SCORE_WATCH threshold
    op_pred = (scores > SCORE_WATCH).astype(int)
    op_fpr  = np.sum((op_pred==1)&(true_labels==0)) / (np.sum(true_labels==0)+1e-9)
    op_tpr  = np.sum((op_pred==1)&(true_labels==1)) / (np.sum(true_labels==1)+1e-9)
    ax.scatter([op_fpr],[op_tpr], color='#ef5350',
               s=80, zorder=5,
               label=f'Operating point (thresh={SCORE_WATCH})')
    ax.set_xlabel('False Positive Rate', color=TC)
    ax.set_ylabel('True Positive Rate', color=TC)
    ax.set_title('ROC Curve', color=TC, fontsize=12)
    ax.legend(facecolor='#1a1a1a', labelcolor=TC, fontsize=9)
    ax.tick_params(colors=TC)
    for sp in ax.spines.values(): sp.set_color(GC)
    plt.tight_layout()
    p2 = os.path.join(save_dir, 'report_2_roc_curve.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.close()
    print(f"  Saved: report_2_roc_curve.png")

    # Optimal threshold (Youden’s J)
    J = tpr - fpr
    ix = np.argmax(J)
    optimal_thresh = thresholds[ix]
    print(f"\nOptimal threshold (Youden J): {optimal_thresh:.3f}")

    # ── 3. SCORE DISTRIBUTION ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9,5), facecolor=BG)
    ax.set_facecolor('#1a1a1a')
    alert_scores  = scores[true_labels==0]
    drowsy_scores = scores[true_labels==1]
    bins = np.linspace(scores.min(), scores.max(), 50)
    ax.hist(alert_scores,  bins=bins, alpha=0.75,
            color=AC, label='Alert',  density=True)
    ax.hist(drowsy_scores, bins=bins, alpha=0.75,
            color=DC, label='Drowsy', density=True)
    ax.axvline(SCORE_WATCH, color='#ffd54f', lw=1.5,
               ls='--', label=f'WATCH threshold ({SCORE_WATCH})')
    ax.axvline(SCORE_WARN,  color='#ff8a65', lw=1.5,
               ls='--', label=f'WARN threshold ({SCORE_WARN})')
    ax.axvline(SCORE_HIGH,  color='#ef5350', lw=1.5,
               ls='--', label=f'HIGH threshold ({SCORE_HIGH})')
    ax.set_xlabel('Fatigue score (robust z-score)', color=TC)
    ax.set_ylabel('Density', color=TC)
    ax.set_title('Score distribution — Alert vs Drowsy',
                 color=TC, fontsize=12)
    ax.legend(facecolor='#1a1a1a', labelcolor=TC, fontsize=9)
    ax.tick_params(colors=TC)
    for sp in ax.spines.values(): sp.set_color(GC)
    plt.tight_layout()
    p3 = os.path.join(save_dir, 'report_3_score_distribution.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.close()
    print(f"  Saved: report_3_score_distribution.png")

    # ── 4. FATIGUE INDEX TIMELINE ──────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16,9),
                             facecolor=BG, sharex=True)
    fig.suptitle('Fatigue Index Timeline — FC5, C3, CP5',
                 color=TC, fontsize=12)
    all_fi = [fi_fc5, fi_c3, fi_cp5]
    ch_cols= ['#ffd54f','#4fc3f7','#ef9a9a']
    trial_nums = np.array([r['trial'] for r in calib_records])

    for i, (ch, fi, col) in enumerate(
            zip(CH_NAMES, all_fi, ch_cols)):
        ax = axes[i]
        ax.set_facecolor('#1a1a1a')
        # Background shading by true label
        for j in range(len(trial_nums)):
            bg = DC if true_labels[j]==1 else AC
            if j < len(trial_nums)-1:
                ax.axvspan(trial_nums[j], trial_nums[j+1],
                           alpha=0.08, color=bg)
        ax.plot(trial_nums, fi, color=col,
                lw=1.0, alpha=0.9, label=ch)
        # Add subject boundaries
        for subj, (start, end) in SUBJECT_BLOCKS.items():
            if start > trial_nums[0]:
                ax.axvline(start, color='#555', lw=0.8,
                           ls=':', alpha=0.7)
                ax.text(start+2, ax.get_ylim()[1]*0.85,
                        f'S{subj}', color='#888',
                        fontsize=6)
        ax.set_ylabel(f'{ch} FI', color=TC)
        ax.legend(facecolor='#1a1a1a', labelcolor=TC,
                  fontsize=8, loc='upper right')
        ax.tick_params(colors=TC)
        for sp in ax.spines.values(): sp.set_color(GC)
    axes[-1].set_xlabel('Trial index', color=TC)
    plt.tight_layout()
    p4 = os.path.join(save_dir, 'report_4_fi_timeline.png')
    plt.savefig(p4, dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.close()
    print(f"  Saved: report_4_fi_timeline.png")

    # ── 5. BAND POWER COMPARISON ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14,5),
                             facecolor=BG)
    fig.suptitle('Band Power — Alert vs Drowsy (FC5)',
                 color=TC, fontsize=12)
    band_data = {
        'theta': theta_fc5,
        'alpha': alpha_fc5,
        'beta' : beta_fc5,
    }
    band_cols = ['#ffd54f','#4fc3f7','#ef9a9a']
    for i, (band, vals) in enumerate(band_data.items()):
        ax = axes[i]
        ax.set_facecolor('#1a1a1a')
        alert_v  = vals[true_labels==0]
        drowsy_v = vals[true_labels==1]
        ax.hist(alert_v,  bins=30, alpha=0.75, color=AC,
                density=True, label='Alert')
        ax.hist(drowsy_v, bins=30, alpha=0.75, color=DC,
                density=True, label='Drowsy')
        # t-test p-value
        t_stat, p_val = stats.ttest_ind(alert_v, drowsy_v)
        change = ((drowsy_v.mean()-alert_v.mean())
                  /alert_v.mean())*100
        ax.set_title(f'{band.capitalize()} power\n'
                     f'{change:+.1f}%  p={p_val:.4f}',
                     color=TC, fontsize=10)
        ax.set_xlabel('Power', color=TC)
        ax.legend(facecolor='#1a1a1a', labelcolor=TC,
                  fontsize=8)
        ax.tick_params(colors=TC)
        for sp in ax.spines.values(): sp.set_color(GC)
    plt.tight_layout()
    p5 = os.path.join(save_dir, 'report_5_band_powers.png')
    plt.savefig(p5, dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.close()
    print(f"  Saved: report_5_band_powers.png")

    # ── 6. PER-SUBJECT ACCURACY ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14,6),
                             facecolor=BG)
    fig.suptitle('Per-Subject Performance',
                 color=TC, fontsize=12)

    subj_ids   = sorted(SUBJECT_BLOCKS.keys())
    subj_acc   = []
    subj_tpr   = []
    subj_fpr   = []
    subj_n     = []

    for s in subj_ids:
        mask = subjects == s
        if mask.sum() < 5:
            subj_acc.append(np.nan)
            subj_tpr.append(np.nan)
            subj_fpr.append(np.nan)
            subj_n.append(0)
            continue
        st = true_labels[mask]
        sp = pred_labels[mask]
        acc = np.mean(st == sp)
        tp  = np.sum((sp==1)&(st==1))
        fn  = np.sum((sp==0)&(st==1))
        fp  = np.sum((sp==1)&(st==0))
        tn  = np.sum((sp==0)&(st==0))
        tpr = tp/(tp+fn+1e-9)
        fpr = fp/(fp+tn+1e-9)
        subj_acc.append(acc)
        subj_tpr.append(tpr)
        subj_fpr.append(fpr)
        subj_n.append(mask.sum())

    # Accuracy bar
    ax = axes[0]; ax.set_facecolor('#1a1a1a')
    colors_acc = ['#4fc3f7' if a >= 0.6 else '#ef5350'
                  if not np.isnan(a) else '#555'
                  for a in subj_acc]
    bars = ax.bar(subj_ids, subj_acc, color=colors_acc,
                  alpha=0.85)
    ax.axhline(0.5, color='#ff8a65', lw=1, ls='--',
               label='Chance (0.5)')
    ax.set_xlabel('Subject', color=TC)
    ax.set_ylabel('Accuracy', color=TC)
    ax.set_title('Accuracy per subject', color=TC,
                 fontsize=10)
    ax.set_xticks(subj_ids)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor='#1a1a1a', labelcolor=TC, fontsize=8)
    ax.tick_params(colors=TC)
    for sp in ax.spines.values(): sp.set_color(GC)
    for bar, n in zip(bars, subj_n):
        if n > 0:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.02,
                    f'n={n}', ha='center',
                    color=TC, fontsize=7)

    # TPR vs FPR scatter
    ax2 = axes[1]; ax2.set_facecolor('#1a1a1a')
    for i, s in enumerate(subj_ids):
        if subj_n[i] > 0 and not np.isnan(subj_tpr[i]):
            ax2.scatter(subj_fpr[i], subj_tpr[i],
                        s=80, color='#ffd54f', zorder=5)
            ax2.annotate(f'S{s}',
                         (subj_fpr[i], subj_tpr[i]),
                         textcoords='offset points',
                         xytext=(5,5), fontsize=7,
                         color=TC)
    ax2.plot([0,1],[0,1], color='#555', lw=1, ls='--')
    ax2.set_xlabel('False Positive Rate', color=TC)
    ax2.set_ylabel('True Positive Rate', color=TC)
    ax2.set_title('TPR vs FPR per subject', color=TC,
                  fontsize=10)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(colors=TC)
    for sp in ax2.spines.values(): sp.set_color(GC)

    plt.tight_layout()
    p6 = os.path.join(save_dir, 'report_6_per_subject.png')
    plt.savefig(p6, dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.close()
    print(f"  Saved: report_6_per_subject.png")

    # ── Print classification report ────────────────────────
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (calibrated trials)")
    print("="*50)
    print(classification_report(
        true_labels, pred_labels,
        target_names=['Alert','Drowsy']))
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Total calibrated trials: {len(calib_records)}")
    print(f"Alert: {(true_labels==0).sum()}  "
          f"Drowsy: {(true_labels==1).sum()}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*52)
    print("EEG FATIGUE — PYTHON ONLY PIPELINE")
    print("="*52)

    print("\nLoading dataset...")
    mat = scipy.io.loadmat(MAT_PATH)
    eeg = mat['EEGsample']          # (2022, 30, 384)
    y   = mat['substate'].ravel()   # (2022,)
    print(f"  Trials:{len(eeg)}  Alert:{(y==0).sum()}  "
          f"Drowsy:{(y==1).sum()}")

    calib  = Calibrator()
    dash   = Dashboard()
    logger = SessionLogger()

    # Store records for end-of-session report
    records = []

    drowsy_count    = 0
    last_alert_time = 0
    prev_score      = 0.0

    print(f"\nPhase 1 — Warmup    : trials 0-{WARMUP_TRIALS-1}")
    print(f"Phase 2 — Calibrate : next {CALIBRATION_TRIALS} "
          f"trials")
    print(f"Phase 3 — Detect    : remaining trials")
    print(f"\nWATCH>{SCORE_WATCH} WARN>{SCORE_WARN} "
          f"HIGH>{SCORE_HIGH} | Confirm:{N_CONFIRM}\n")

    hdr = (f"{'':5} {'Trial':>6} | {'Score':>7} | "
           f"{'FC5_FI':>7} {'C3_FI':>7} {'CP5_FI':>7} | "
           f"{'Lvl':<5} {'True':<7} {'Subj'} Status")
    print(hdr); print("-"*len(hdr))

    try:
        for trial_idx in range(len(eeg)):

            # Preprocess 3 channels
            trial_raw  = eeg[trial_idx, CH_IDX, :].astype(float)
            trial_filt = np.zeros_like(trial_raw)
            for i in range(3):
                trial_filt[i] = preprocess(trial_raw[i])

            # Band powers + fatigue index
            bpowers   = {}
            fi_per_ch = {}
            theta_alpha_per_ch = {}
            for i, ch in enumerate(CH_NAMES):
                bp = compute_band_powers(trial_filt[i])
                bpowers[ch]   = bp
                fi_per_ch[ch] = fatigue_index(bp)
                theta_alpha_per_ch[ch] = theta_alpha_ratio(bp)

            # Calibration
            calib.update(fi_per_ch, trial_idx)
            calib_status = calib.status(trial_idx)

            # Score with EMA smoothing
            raw_score = compute_score(
                fi_per_ch, calib.medians, calib.iqr_stds)

            if trial_idx == 0:
                score = raw_score
            else:
                score = EMA_ALPHA * raw_score + (1 - EMA_ALPHA) * prev_score
            prev_score = score

            # Decision
            if not calib.ready or trial_idx < WARMUP_TRIALS:
                drowsy_count = 0
                level        = 'OK'
            elif score > SCORE_WARN:
                drowsy_count += 1
                if drowsy_count >= 3:
                    level = 'HIGH'
                else:
                    level = 'WARN'
            elif score > SCORE_WATCH:
                drowsy_count += 1
                level = 'WATCH'
            else:
                drowsy_count = 0
                level        = 'OK'

            true_label = int(y[trial_idx])
            subject    = get_subject(trial_idx)

            # Console
            tag = {'HIGH':'[!!]','WARN':'[! ]',
                   'WATCH':'[~  ]','OK':'[   ]'}.get(level,'[?]')
            print(f"{tag} {trial_idx:6d} | "
                  f"{score:+7.3f} | "
                  f"{fi_per_ch['FC5']:7.3f} "
                  f"{fi_per_ch['C3']:7.3f} "
                  f"{fi_per_ch['CP5']:7.3f} | "
                  f"{level:<5} "
                  f"{'DROWSY' if true_label==1 else 'ALERT '} "
                  f"S{subject} {calib_status}")

            # Telegram
            if level in ('HIGH','WARN'):
                now = time.time()
                if now - last_alert_time > COOLDOWN_SEC:
                    send_telegram(
                        f"FATIGUE ALERT — {level}\n"
                        f"Trial  : {trial_idx}\n"
                        f"Subject: {subject}\n"
                        f"Score  : {score:.3f}\n"
                        f"FC5 FI : {fi_per_ch['FC5']:.3f}\n"
                        f"C3  FI : {fi_per_ch['C3']:.3f}\n"
                        f"CP5 FI : {fi_per_ch['CP5']:.3f}")
                    last_alert_time = now

            # Store record
            records.append({
                'trial'       : trial_idx,
                'subject'     : subject,
                'score'       : score,
                'fi_FC5'      : fi_per_ch['FC5'],
                'fi_C3'       : fi_per_ch['C3'],
                'fi_CP5'      : fi_per_ch['CP5'],
                'theta_FC5'   : bpowers['FC5']['theta'],
                'alpha_FC5'   : bpowers['FC5']['alpha'],
                'beta_FC5'    : bpowers['FC5']['beta'],
                'level'       : level,
                'true_label'  : true_label,
                'calib_status': calib_status,
            })

            # Log
            logger.log(trial_idx, subject, score,
                       fi_per_ch, bpowers['FC5'],
                       level, true_label, calib_status)

            # Dashboard
            dash.update(trial_filt, bpowers, fi_per_ch,
                        score, level, true_label,
                        calib_status, calib, subject)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        logger.close()
        plt.ioff()
        # Generate full analysis report
        generate_report(records, SAVE_DIR)
        plt.show()

if __name__ == "__main__":
    main()