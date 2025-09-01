# Predicting Four Canonical Odor Qualities from Molecular Structure
**Task:** given a molecule’s SMILES, predict whether it is **pungent**, **sweet**, **floral**, and/or **minty**.
---
## Abstract
>I predict four canonical odor qualities - **pungent**, **sweet**, **floral**, **minty** - directly from molecular 
> structure. Free-text odor descriptors from _pyrfume-data_ are mapped to these targets using a sentence-transformer 
> with label-specific thresholds, while molecules are represented by Mordred descriptors computed from SMILES. Three 
> classical baselines (**Logistic Regression**, **Random Forest**, **XGBoost**) are trained in a **One-vs-Rest** setup 
> with per-label **F1-optimized thresholds**. On the held-out test set, **XGBoost** provides the best overall 
> trade-off (higher **PR-AUC** and **F1**), and **Logistic Regression** yields the highest **recall**. Results align with 
> known effects of class imbalance - `sweet`/`floral` outperform `pungent`/`minty` in PR-AUC, suggesting further gains 
> from calibration, label-correlation modeling, and broader baselines such as **Mol-PECO**.
---
## 1) Dataset & Labels (what went in)
- **Source curation:** Human odor datasets aggregated from _pyrfume-data_ were normalized and merged by SMILES.
- **Free-text → 4 labels:** The free-form descriptors were embedded with a sentence transformer (`multi-qa-MiniLM-L6-cos-v1`). For each molecule, cosine similarities to the four target strings were computed and **label-specific thresholds** were applied to produce binary labels:
  - **Thresholds used:** `pungent=0.395`, `sweet=0.345`, `floral=0.385`, `minty=0.505`. 
  - **Output file:** `data/molecule_scent_labels_with_probs.csv` with both probabilities (`prob_*`) and hard labels (`label_*`).
- **Features:** Mordred (2D) descriptors computed per SMILES (after RDKit sanitization), numeric columns retained.
  - Typical shape in runs here: **~8.7k molecules × ~700 features** (after dropping non-numeric).
- **Split:** multi-label-aware stratification (stratifying on the per-row label “signature”) into **train / val / test = 80 / 10 / 10** with a fixed seed.
---
## 2) Models (what was trained)
All are **One-vs-Rest** multi-label classifiers trained on the Mordred features:
- **Logistic Regression (LR):** median imputation → standardization → `LogisticRegression(max_iter=1000, class_weight="balanced")`.
- **Random Forest (RF):** median imputation → `RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=17)`.
- **XGBoost (XGB):** median imputation → `XGBClassifier` (tree_method=`hist`, `n_estimators=400`, typical defaults).
Thresholding: Per-label decision thresholds were **optimized on the validation set to maximize F1**, then applied to test probabilities.
---
## 3) Metrics (how it was measured)
- **Macro ROC-AUC:** threshold-independent ranking quality (averaged over the four labels).
- **Macro PR-AUC:** precision-recall area averaged over labels; **preferred under class imbalance**.
- **Macro F1, Precision, Recall:** after applying tuned thresholds.
- **Subset accuracy (exact match):** fraction of molecules where **all four labels** are predicted exactly right. (Reported in the notebook when requested; not the primary model selector due to sensitivity to rare labels.)
---
## 4) Results (what came out)
Below is a concise comparison based on the latest runs visible in the notebook outputs.

| Model             |                          ROC-AUC |                           PR-AUC |                               F1 |                       Precision |                          Recall |
| ----------------- |---------------------------------:|---------------------------------:|---------------------------------:|--------------------------------:|--------------------------------:|
| **LogReg**        |                            0.765 |                            0.581 |                            0.613 |                           0.516 |                       **0.780** |
| **Random Forest** |                            0.770 |                            0.604 |                            0.611 |                           0.522 |                           0.761 |
| **XGBoost**       |                        **0.782** |                        **0.617** |                        **0.619** |                       **0.540** |                           0.739 |
#### What this says:
- **RF** edges out **LR** on **PR-AUC**, and **precision**, with a small advantage in **ROC-AUC** as well.
- **LR** delivers the **highest recall**, which is typical for linear models under class imbalance with class weighting + standardization.
- **XGBoost:** in most cheminformatics classification tasks XGB is often **on par with or slightly ahead of RF** on ROC-AUC / PR-AUC when its depth, learning rate, and subsampling are tuned. The same is evident here, as XGB delivers the **best values in all parameters but Recall**. 
> **Takeaway:** For the current thresholds and feature set, XGB is the best overall trade-off (ROC-AUC/PR-AUC/F1/precision), while LR is preferable if max recall is the priority (e.g., to minimize false negatives in screening).

---
## 5) Per-label behavior & class imbalance (what it means)
In this corpus, `sweet` and `floral` labels are more prevalent than `pungent` and `minty` (common in human-curated scent descriptors). This imbalance affects metrics and curves:
- **PR-AUC sensitivity:** PR-AUC depends on class prevalence. With **fewer positives**, the **baseline precision** is lower, depressing PR-AUC even when the ranking quality (ROC-AUC) is decent. This explains why **pungent/minty** often show **lower PR-AUC** than sweet/floral, while ROC-AUC gaps are smaller. 
- **Threshold tuning trade-offs:** Optimizing F1 per label on the validation set helps balance precision/recall. For rare labels, **recall can remain fragile:** small threshold changes lead to big swings in precision due to few positives. 
- **Model patterns observed:**
  - **LR** tends to push **higher recall** at the expense of precision - helpful for rare labels but leads to more false positives. 
  - **RF/XGB** typically achieve **higher precision** (and PR-AUC), meaning **fewer false positives**, but may miss some true positives without careful thresholding.

#### Mitigations to consider (future work):
- Adjust thresholds per label to meet application-specific precision/recall targets (maximize F-β, or fix precision ≥ X%). 
- Use **class-balanced focal loss** (if moving to neural models) or **calibrated resampling** (e.g., SMOTE-variants) with caution. 
- **Classifier chains** or **label-correlation models** to exploit co-occurrence of descriptors (e.g., sweet ↔ fruity proxies).

---
## 6) Where this sits vs. published pipelines (context)
This pipeline follows well-established ideas in the olfaction/cheminformatics literature:
- **Feature-based prediction of percepts:** e.g., predicting human olfactory perception from chemical features ([_Science_, 2017](https://doi.org/10.1126/science.aal2014)) and subsequent multi-label modeling of perceptual descriptors ([Nat Commun, 2018](doi:10.1038/s41467-018-07439-9)).
- **Graph-/physchem-based baselines:** **Mol-PECO** ([_npj Syst Biol Appl_, 2024](doi:10.1038/s41540-024-00401-0)) is a modern alternative baseline combining curated odors, molecular representations, and task-specific training. It could be wired in as a **no-code CLI** baseline if the scope allowed (and mapped to the four canonical labels), providing an orthogonal comparison to Mordred-based models.

---
## 7) Limitations & next steps (what I’d do next)
- **Calibration:** apply Platt/Isotonic calibration per label (on validation data) to improve probability quality, then re-optimize thresholds. 
- **Label mapping:** expand the synonym sets and verify the sentence-embedding thresholds on a hand-labeled subset. 
- **Cross-validation reporting:** include fold distributions (PR-AUC/ROC-AUC) and confidence intervals; bootstrap per-label metrics. 
- **Modeling label correlations:** classifier chains, low-rank label embeddings, or sequence models over descriptor lists. 
- **External validity:** evaluate on strictly held-out datasets or by **dataset-wise splits** to assess domain shift.
---
## 8) Reproducibility (how to rerun)
- **Environment:** environment.yml pins Python 3.10 and package versions compatible with RDKit, Mordred/MordredCommunity, XGBoost, and sentence-transformers.
- **Determinism:** fixed random seed (17).
#### Notebook flow:
0) Setup – create env, set paths.
1) Build labels – run the sentence-transformer step to produce molecule_scent_labels_with_probs.csv.
2) Mordred features – compute/read features/mordred.parquet.
3) Multi-label split – stratified train/val/test.
4) Train baselines – LR, RF, XGB with per-label threshold tuning; save metrics and curves under artifacts/classical/.
---
## References
- **Castro et al.** Pyrfume: A Window to the World’s Olfactory Data. _BioRiv_ (2022). **doi:10.1101/2022.09.08.507170**
- **Keller et al.** Predicting human olfactory perception from chemical features of odor molecules. _Science_ (2017). **doi:10.1126/science.aal2014**
- **Gutiérrez et al.** Predicting natural language descriptions of mono-molecular odorants. _Nat Commun_ (2018). **doi:10.1038/s41467-018-07439-9**
- **Zhang et al.** A deep position-encoding model for predicting olfactory perception from molecular structures and electrostatics. _npj Systems Biology and Applications_ (2024). **doi:10.1038/s41540-024-00401-0**

