# 5GNID: 5G Network Intrusion Detection Pipeline

End-to-end machine learning pipeline for intrusion detection on the 5G-NIDD dataset, with a systems focus on accuracy, throughput, latency, and uncertainty handling.

## Project Highlights

- Multi-stage cascade IDS design for real-time 5G traffic.
- Baselines included: TabNet and LCCDE ensemble.
- Cascade variants evaluated: Sequential Binary-First, Sequential Multiclass-First, Parallel Voting, Parallel Confidence.
- Best reported result: ExtraTrees cascade with ~0.999666 accuracy on known predictions and 0.19% uncertainty isolation.

## Repository Structure

- `5gnidd_full_pipeline_Final.ipynb`: ExtraTrees-based full pipeline (best model experiments).
- `5gnidd_lccde_tabnet_colab.ipynb`: TabNet and LCCDE baseline experiments.
- `5gnidd_rf_full_pipeline.py`: End-to-end RandomForest cascade script.
- `Data/Combined.csv`: dataset file (large file; use Git LFS for GitHub).
- `Output/EToutput.txt`: ExtraTrees experiment output logs.
- `Output/RFoutput.txt`: RandomForest experiment output logs.
- `Output/LCCDE_TABNET_OUTPUT.txt`: TabNet/LCCDE output logs.
- `ResearchPaper.md`: conference-style draft write-up.

## Methods Implemented

1. Binary anomaly detection (`Normal` vs `Attack`).
2. Multiclass attack classification (`Benign`, `UDPFlood`, `HTTPFlood`, `SlowrateDoS`, `TCPConnectScan`, `SYNScan`, `UDPScan`, `SYNFlood`, `ICMPFlood`).
3. Routing/fusion strategies:
   - Sequential Binary-First
   - Sequential Multiclass-First
   - Parallel Voting
   - Parallel Confidence (with unknown/suspicious routing)

## Data Processing Pipeline

- Raw traffic preprocessing and schema normalization.
- Feature subsets for binary and multiclass stages.
- Class imbalance handling with hybrid resampling:
  - RandomUnderSampler
  - SMOTE
- Stratified train/test evaluation.

## Reported Results (from logs)

### Baselines (`Output/LCCDE_TABNET_OUTPUT.txt`)

| Model | Accuracy | Macro F1 | Weighted F1 | Inference Time (s) | Throughput (samples/s) |
|---|---:|---:|---:|---:|---:|
| TabNet | 0.92542 | 0.568486 | 0.930409 | 4.195169 | 99,135.47 |
| LCCDE | 0.75286 | 0.600625 | 0.785129 | 196.760732 | 2,113.68 |

### ExtraTrees Cascade (`Output/EToutput.txt`, test size 243,178)

| Strategy | Accuracy (known only) | Macro F1 | Throughput (samples/s) | Unknown % |
|---|---:|---:|---:|---:|
| Sequential Binary-First | 0.999231 | 0.998303 | 22,521.26 | 0.00 |
| Sequential Multiclass-First | 0.999124 | 0.998146 | 27,930.53 | 0.00 |
| Parallel Voting | 0.999124 | 0.998146 | 12,860.58 | 0.00 |
| Parallel Confidence | **0.999666** | **0.999402** | 14,019.70 | **0.19** |

### RandomForest Cascade (`Output/RFoutput.txt`, test size 415,890)

| Strategy | Accuracy (known only) | Throughput (samples/s) | Time/sample (ms) | Unknown % |
|---|---:|---:|---:|---:|
| Sequential Binary-First | 0.9958 | 191,951.19 | 0.00521 | 8.8 |
| Sequential Multiclass-First | 0.9957 | 279,470.24 | 0.00358 | 8.8 |
| Parallel Voting | 0.9957 | 183,039.65 | 0.00546 | 8.8 |
| Parallel Confidence | **0.9966** | 173,594.97 | 0.00576 | 24.7 |

## Environment Setup

Use Python 3.10+ (recommended 3.10 or 3.11).

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy pandas scikit-learn imbalanced-learn joblib
```

For notebook baselines, you may also need:

```bash
pip install xgboost lightgbm catboost pytorch-tabnet torch seaborn matplotlib jupyter
```

## How To Run

### A) RandomForest pipeline script

```bash
python 5gnidd_rf_full_pipeline.py
```

Important: verify dataset path inside the script.

- Current code path is set to `Combined.csv/Combined.csv` relative to script.
- In this repository, dataset is at `Data/Combined.csv`.
- Update `RAW_CSV_PATH` in `5gnidd_rf_full_pipeline.py` if needed.

### B) Notebooks

- Open and run all cells in:
  - `5gnidd_full_pipeline_Final.ipynb` (ExtraTrees pipeline)
  - `5gnidd_lccde_tabnet_colab.ipynb` (TabNet + LCCDE baselines)

Some notebook cells are Colab-oriented (`google.colab` imports). If running locally, remove or adapt those mount cells.

## Reproducibility Notes

- Keep train/test split stratified.
- Apply hybrid resampling only on training split.
- Keep test set untouched to preserve natural rare-class distribution.
- Set random seeds where available (`RANDOM_STATE = 42`).

## GitHub Push and Large Files

`Data/Combined.csv` is larger than GitHub's 100 MB file limit. Track it with Git LFS before push:

```bash
git lfs install
git lfs track "Data/Combined.csv"
git add .gitattributes Data/Combined.csv
git commit -m "Track dataset with Git LFS"
git push
```

If the file is already in commit history, migrate it:

```bash
git lfs migrate import --include="Data/Combined.csv" --include-ref=refs/heads/main
git push --force origin main
```

## Citation

If you use this project, cite the accompanying research draft in `ResearchPaper.md` and the 5G-NIDD dataset source referenced in the code.
