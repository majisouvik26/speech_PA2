# Speech Understanding Assignment 2

## Overview
This repository contains solutions for two speech processing tasks:  
1. **Speech Enhancement** including speaker verification, separation, and identification.  
2. **MFCC-based Language Classification** for 10 Indian languages.  

---

## Task 1: Speech Enhancement

### Key Features:
- Pre-trained WavLM model fine-tuned using LoRA.
- SepFormer used for speech separation.
- Multi-speaker dataset synthesis pipeline.

### Results:

| Metric                             | Pre-trained | Fine-tuned |
|------------------------------------|-------------|------------|
| EER (%)                            | 36.73       | 19.73      |
| TAR @ 1% FAR (%)                   | 7.39        | 24.39      |
| Speaker Identification Rank‑1 (%) | 36.11       | 54.31      |
| Verification Accuracy (%)         | 66.37       | 75.87      |

---

## Task 2: Language Classification

### Key Features:
- MFCC extraction and visualization tools.
- Random Forest classifier achieving ~98% accuracy.

### Results:

**Classification Accuracy:** `0.98`

```
Classification Report:
              precision    recall  f1-score   support

     Bengali       0.96      0.99      0.98      5451
    Gujarati       0.96      0.94      0.95      5287
       Hindi       0.98      0.99      0.99      5092
     Kannada       1.00      0.95      0.97      4442
   Malayalam       0.98      0.98      0.98      4808
     Marathi       0.99      0.99      0.99      5075
     Punjabi       0.95      0.96      0.95      5244
       Tamil       0.99      0.99      0.99      4838
      Telugu       0.99      0.98      0.98      4731
        Urdu       0.97      0.99      0.98      6392

    accuracy                           0.98     51360
```

---

### Usage:

1. Run `task_1/download_scripts/dataset.py` to download datasets.
2. Execute `task_1/ftune_wavelm/main.py` for fine-tuning.
3. Use `task_2/task_a.py` and `task_2/task_b.py` for MFCC analysis and classification.

---

## Directory Structure

```
majisouvik26-speech_pa2/
├── README.md
├── LICENSE
├── task_1/
│   ├── data.py
│   ├── evaluation_results.txt
│   ├── trial_pair_eval.py
│   ├── wavlm.py
│   ├── download_scripts/
│   │   ├── clean_download.py
│   │   └── dataset.py
│   ├── ftune_wavelm/
│   │   ├── __init__.py
│   │   ├── arcface_loss.py
│   │   ├── check.py
│   │   ├── dataset.py
│   │   ├── evaluation.py
│   │   ├── main.py
│   │   ├── model_utils.py
│   │   ├── results.txt
│   │   └── train.py
│   └── multi_speaker/
│       ├── __init__.py
│       ├── dataset_creation.py
│       ├── identification.py
│       ├── main.py
│       ├── mixing.py
│       ├── requirements.txt
│       ├── separation.py
│       ├── separation_evaluation.py
│       ├── task_1_separation.py
│       └── task_2_identification.py
└── task_2/
    ├── classification_results.txt
    ├── config.py
    ├── dataloader.py
    ├── feature_extraction.py
    ├── model.py
    ├── requirements.txt
    ├── stats.py
    ├── task_a.py
    ├── task_b.py
    ├── visualization.py
    └── plots/
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/majisouvik26/speech_PA2?tab=MIT-1-ov-file) file for details.
