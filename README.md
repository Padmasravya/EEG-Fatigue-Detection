# EEG-Based Driver Fatigue Detection

This project implements a real-time EEG-based driver fatigue detection system using digital signal processing and adaptive calibration.

## Features
- EEG preprocessing (DC removal, bandpass, notch)
- Welch PSD-based feature extraction
- Fatigue Index: theta / (alpha + beta)
- Adaptive per-subject calibration (median + IQR)
- Real-time scoring with EMA smoothing
- Alert system with Telegram integration
- Live visualization dashboard

## Dataset
Figshare EEG Driver Drowsiness Dataset  
11 subjects, 2022 trials, 30 channels, 128 Hz

## Files
- `train_model.py` → Training and feature extraction
- `stream_eeg.py` → Real-time inference system
- `model_profile.json` → Learned parameters

## Results
- Accuracy: 61%
- F1 Score: 0.60
- AUC: 0.64
