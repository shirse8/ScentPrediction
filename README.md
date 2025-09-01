# ScentPrediction — Coarse Olfactory Classification from Molecular Structure

**Goal.** Predict whether a molecule smells **pungent**, **sweet**, **floral**, and/or **minty** from structure alone.

This repo demonstrates a **reproducible** pipeline:
1. **Data curation** from `pyrfume-data` → consolidated SMILES + free-form descriptors.
2. **Semantic mapping** of descriptors to four coarse labels using a **sentence-transformer**.
3. **Feature engineering** with **Mordred 2D descriptors** (RDKit).
4. **Multi-label modeling** with classical baselines (LogReg, RandomForest, XGBoost).
5. **Transparent evaluation** (ROC/PR curves, per-label and macro metrics), with **validation-tuned thresholds**.

---

## Quickstart

### 0) Create the environment
```bash
conda env create -f environment.yml
conda activate scentlab
python -m ipykernel install --user --name scentlab --display-name "Python (scentlab)"
```

### 1) Open the notebook
Run `ScentPrediction_Playbook.ipynb` top-to-bottom.
All heavy steps cache outputs so iterative runs are fast. 
- If `data/labeled_molecules_from_human_odor_datasets.csv` is missing, the notebook automatically builds it from pyrfume-data (no manual downloads). 
- The sentence-transformer step writes `data/molecule_scent_labels_with_probs.csv` (keeps per-label probabilities and hard labels). 
- Mordred features cache to `features/mordred.parquet`.
Artifacts (plots + metrics) are saved to `artifacts/classical/`.

---

## Repository layout
```bash
data/
  labeled_molecules_from_human_odor_datasets.csv     # consolidated (auto-built if missing)
  molecule_scent_labels_with_probs.csv               # encoder-derived coarse labels
features/
  mordred.parquet                                    # cached Mordred 2D descriptors
artifacts/
  classical/                                         # metrics tables + ROC/PR plots
scripts/
  prepare_human_datasets.py                          # build consolidated dataset from pyrfume-data
  features.py                                        # Mordred cache builder
  train_baselines.py                                 # CLI to run classical baselines end-to-end
  utils.py                                           # split/eval utilities
ScentPrediction_Playbook.ipynb                       # end-to-end, well-explained workflow
environment.yml                                      # pinned environment (conda-forge)
```

---

## Method overview
**Semantic mapping (labels).** Free-text descriptors (e.g., “pear”, “spearmint”, “rose”) are embedded with a 
**sentence-transformer** (`multi-qa-MiniLM-L6-cos-v1`) and matched to the four target concepts by 
**cosine similarity**, followed by **label-specific thresholds** tuned via manual validation. This yields robust 
binary targets while preserving per-label probabilities.

**Features.** Compute Mordred 2D descriptors from SMILES using RDKit; descriptors are numeric-only and cached to Parquet.

**Models.** Train One-Vs-Rest classifiers for each label:
- Logistic Regression (standardized; class-balanced)
- Random Forest (class-balanced)
- XGBoost (non-linear baseline)

**Evaluation.**
- **ROC/PR curves** with **AUCs in the legends**, per label 
- **Macro** metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)
- **Thresholds** are optimized on the validation set (per label, maximize F1), then **frozen** for the Test set.

---

## Reproducibility
- Deterministic splits (seeded). 
- All expensive steps cached (`features/mordred.parquet`, labeled CSVs).
- Environment is fully pinned (`conda-forge`).

---

## Acknowledgements
- [pyrfume](https://github.com/pyrfume) for curated odor datasets. 
- [RDKit](https://www.rdkit.org/) and [mordredcommunity](https://pypi.org/project/mordredcommunity/) for cheminformatics. 
- [Sentence-Transformers](https://www.sbert.net/) for efficient text embeddings.
