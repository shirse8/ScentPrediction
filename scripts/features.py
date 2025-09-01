from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from tqdm import tqdm


def build_mordred_cache(data_csv: Path, out_path: Optional[Path] = None, smiles_col: str = "SMILES") -> pd.DataFrame:
    """
    Compute Mordred (2D) descriptors and cache to parquet.
    Returns the DataFrame (index = SMILES).
    """
    # Resolve input/output paths and ensure output directory exists
    data_csv = Path(data_csv)
    out_path = out_path or (data_csv.parent.parent / "features" / "mordred.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input and validate presence of the SMILES column
    df = pd.read_csv(data_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Missing '{smiles_col}' column in {data_csv}")
    smiles = df[smiles_col].astype(str).str.strip().tolist()

    # Initialize Mordred calculator (2D only)
    calc = Calculator(descriptors, ignore_3D=True)

    # Parse SMILES to RDKit molecules (keep track of parseability)
    mols = []
    keep = []
    for s in tqdm(smiles, desc="RDKit parse"):
        m = Chem.MolFromSmiles(s)
        if m is None:
            mols.append(None)
            keep.append(False)
        else:
            mols.append(m)
            keep.append(True)

    # Compute descriptors; skip molecules that raise errors in Mordred
    feats = []
    kept_smiles = []
    for s, m in tqdm(list(zip(smiles, mols)), desc="Mordred", total=len(smiles)):
        if m is None:
            continue
        try:
            v = calc(m)
            feats.append(pd.Series(v.asdict()))
            kept_smiles.append(s)
        except Exception:
            # skip if Mordred fails
            continue

    # Build feature table indexed by SMILES
    X = pd.DataFrame(feats, index=kept_smiles)

    # Keep only numeric columns
    num_cols = X.select_dtypes(include=["number"]).columns
    X = X[num_cols]

    # Persist to parquet and report shape
    X.to_parquet(out_path, index=True)
    print(f"✔ Mordred cache: {out_path}  (rows={len(X)}, cols={len(X.columns)})")
    return X


if __name__ == "__main__":
    # Minimal CLI: read CSV with SMILES and write descriptors parquet
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=Path, required=True, help="CSV with at least a SMILES column")
    ap.add_argument("--out_path", type=Path, default=None, help="optional output parquet path")
    args = ap.parse_args()
    build_mordred_cache(args.data_csv, args.out_path)
