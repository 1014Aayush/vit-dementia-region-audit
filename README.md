# Accuracy Is Not Reliance

Code for *Accuracy Is Not Reliance: Auditing Which Brain Regions Frozen Vision Transformers Need to Detect Dementia* (A. Pokhrel, 2026).

## Pipeline
| Step | Code |
|---|---|
| Extract OASIS-1 T88 images; region masks | `scripts/oasis_masks.py extract`, `scripts/oasis_masks.py masks` |
| SynthSeg segmentation | `notebooks/oasis_synthseg.ipynb` |
| Slice/crop geometry (label-blind) | `scripts/mask_geometry.py`, `scripts/mask_geometry_options.py` |
| Cohort, folds, volumetric baseline | `scripts/baseline.py` (shared settings in `scripts/common.py`) |
| ViT input slices | `scripts/make_slices.py` |
| ViT features under attention cuts | `notebooks/oasis_vit_features.ipynb` (uses `scripts/vitprobe/`) |
| Gate, lesion tests, statistics | `scripts/analyze_vit.py`, `scripts/protocol_a.py`, `scripts/l1_probe.py` |
| Positive control | `scripts/positive_control_features.py`, `notebooks/oasis_positive_control.ipynb`, `scripts/positive_control_analysis.py` |

## Data
This repository contains code only. The analyses use OASIS-1 (Marcus et al., 2007), which must be obtained from https://sites.wustl.edu/oasisbrains/ under its data use terms. No images, segmentations or per-subject data are included.

## Requirements
Python 3.12 with `requirements.txt`; the notebooks ran on Kaggle (timm 1.0.30). SynthSeg 2.0 needs Python 3.8 + TensorFlow 2.2 (set up inside the notebook).
