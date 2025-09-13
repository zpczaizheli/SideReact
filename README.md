# SideReact

## Title
Interpretable Side Reaction Detection via Atom Representations Derived from Multi-tier Reaction Perspectives

## Overview
<img width="3248" height="4744" alt="Model-overview" src="https://github.com/user-attachments/assets/b41c8271-041a-4182-8cec-c1f3adb9b46c" />


## Environment Requirements
- Python = 3.7.8
- numpy = 1.21.6
- pytorch = 1.13.1
- rdkit = 2022.03.1
- Matplotlib = 3.5.1


## Process

### Step 0 — Project root

Start from the repository root. Example:

```bash
cd SideReact  
```

### Step 1 — Data Processing

The goal of this step is to produce the feature files required by the model.

#### 1.1 Generate motif embeddings

Run the motif extraction script:

```bash
python Motif_Feature/motif.py
```

This will produce:

* `motif_embeddings.xlsx`

After the script finishes, **copy** (or move) `motif_embeddings.xlsx` into the parent `Data/` folder


#### 1.2 Generate atom-level features for reactions

Run the reaction atom-feature extraction script:

```bash
python Reaction_feature/get_atom_feature.py
```

This will produce:

* `Data_reaction_atom_features_reactants.xlsx`

Place (or confirm) this file is located in the `Data/` folder. The file contains per-atom embedding vectors for reactants across all processed reactions.

> **Note:** Deployment and training details for the large model **Qwen** (if used as part of feature generation) are documented in `Reaction_feature/detail.txt`. Consult that file for environment, GPU, and training specifics related to Qwen.

---

### Step 2 — Model Execution

After data files are ready in `Data/`, run the main model script:

```bash
python model.py
```

Expected output:

* `model_OUT.txt` — model performance / prediction results (text file)

`model.py` will read `Data/motif_embeddings.xlsx` and `Data/Data_reaction_atom_features_reactants.xlsx` (or the exact file paths coded in the script). If your `model.py` expects different filenames or paths, edit the script configuration accordingly.


-
---

## Troubleshooting

* If `motif.py` or `get_atom_feature.py` fails with missing package errors, install the indicated package (see **Environment Requirements**).
* If `model.py` cannot find the data files, verify the filenames and locations in the script, or update `model.py` to point to the correct `Data/` file paths.
* For Qwen-related issues, check `Reaction_feature/detail.txt` for GPU, token, or configuration settings used during feature generation.

---



