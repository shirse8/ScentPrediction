from typing import Optional, Iterable, Tuple, List
import re
import json
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm
from pyrfume import load_data


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/master/"

# Descriptor columns (case-insensitive substrings)
DESCRIPTOR_COLS = [
    "labels", "filtered descriptors", "odor percepts",
    "descriptors", "descriptor 1", "descriptor 2", "descriptor 3",
    "primary odor", "sub odor",
]

# Columns that identify stimuli / molecules
ID_COLS = ["stimulus"]
CID_COLS = ["cid", "new_cid", "cids"]
SMILES_COLS = ["smiles", "isomeric_smiles", "isomericsmiles", "canonicalsmiles"]

# behavior file overrides
BEHAVIOR_FILE_MAP = {
    "arctander_1960": "behavior_1_sparse.csv",
    "leffingwell": "behavior_sparse.csv",
    "sharma_2021b": "behavior_1.csv",
    "sigma_2014": "behavior_sparse.csv",
}


# ---------------------------------------------------------------------
# Inventory parsing
# ---------------------------------------------------------------------
def get_inventory() -> str:
    """Download pyrfume-data inventory markdown as text."""
    url = BASE_URL + "tools/inventory.md"
    r = requests.get(url)
    r.raise_for_status()
    return r.text

def parse_badges(block: str) -> dict:
    """Extract badge key/value pairs from an inventory line."""
    badges = {}
    for m in re.finditer(r"label=([^&]+)&message=([^&]+)", block):
        label = m.group(1).strip().lower()
        message = m.group(2).strip()
        badges[label] = message
    return badges

def filter_human_odor_datasets(inventory_text: str) -> list[tuple[str, dict]]:
    """Return (dataset, badges) for human, non-mixture odorCharacter datasets."""
    datasets = []
    for block in inventory_text.split("<br>"):
        block = block.strip()
        if not block or not block.startswith("[!["):
            continue
        ds_m = re.search(r"\[\!\[(.*?)\]", block)
        if not ds_m:
            continue
        dataset = ds_m.group(1).strip()
        badges = parse_badges(block)
        if (
            badges.get("organism", "").lower() == "human"
            and badges.get("data", "").lower() == "odorcharacter"
            and badges.get("stimuli", "").lower() != "mixtures"
        ):
            datasets.append((dataset, badges))
    return datasets


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_csv(dataset: str, fname: str) -> pd.DataFrame:
    """Load a CSV via pyrfume.load_data; return empty DataFrame if missing (no logging here)."""
    try:
        return load_data(f"{dataset}/{fname}")
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _col_in_list(col: str, candidates: Iterable[str]) -> bool:
    """Case-insensitive substring match against candidate tokens."""
    c = col.lower()
    return any(key in c for key in candidates)

def find_id_column(df: pd.DataFrame, candidates: list) -> Tuple[pd.DataFrame, Optional[str]]:
    """Find a matching ID-like column or index; return possibly-reset df and column name."""
    for col in df.columns:
        if _col_in_list(col, candidates):
            return df, col
    if df.index.name and _col_in_list(df.index.name, candidates):
        new_col = df.index.name
        df = df.reset_index().rename(columns={new_col: new_col})
        return df, new_col
    if _col_in_list("index", candidates):
        df = df.reset_index().rename(columns={"index": "index"})
        return df, "index"
    return df, None

def find_any(df: pd.DataFrame, candidates: list) -> list:
    """Return all columns whose names match any candidate substring."""
    return [c for c in df.columns if _col_in_list(c, candidates)]

def normalize_cid_series(s: pd.Series) -> pd.Series:
    """Normalize CIDs to first numeric token (digits-only); allow empty."""
    if s is None or s.empty:
        return s
    s2 = s.astype(str).str.split(r"[;,]").str[0].str.extract(r"(\d+)", expand=False)
    return s2

def normalize_id_series(s: pd.Series) -> pd.Series:
    """Normalize ID-like series to stripped strings, preserving integers."""
    if s is None or s.empty:
        return s
    if pd.api.types.is_numeric_dtype(s):
        s2 = s.astype("Int64").astype("string")
    else:
        s2 = s.astype(str)
    return s2.str.strip()

def is_all_bool_or_numeric(df: pd.DataFrame, cols: list) -> bool:
    """True if every selected column is boolean or numeric."""
    if not cols:
        return True
    sub = df[cols]
    return all(pd.api.types.is_bool_dtype(sub[c]) or pd.api.types.is_numeric_dtype(sub[c]) for c in sub.columns)

def parse_descriptor_cell(val) -> list[str]:
    """Parse a cell into a list of descriptor tokens; map 'bland'→'odorless'."""
    if isinstance(val, list):
        out = []
        for x in val:
            t = _clean_desc_token(x)
            if t:
                out.append(t)
        return out
    if isinstance(val, str):
        parts = re.split(r"[;,/]", val)
        out = []
        for p in parts:
            t = _clean_desc_token(p)
            if t:
                if t == "bland":
                    t = "odorless"  # necessary fix since my chosen model confuses a "bland" descriptor as sweet
                out.append(t)
        return out
    return []

def extract_descriptor_lists(df: pd.DataFrame) -> pd.Series:
    """Collect descriptor columns per row; return list-of-strings (or sentinel if all numeric/bool)."""
    cols = [c for c in df.columns if _col_in_list(c, DESCRIPTOR_COLS)]
    if not cols:
        return pd.Series([[]] * len(df), index=df.index)
    if is_all_bool_or_numeric(df, cols):
        return pd.Series([["__ALL_NUMERIC_OR_BOOL__"]] * len(df), index=df.index)
    out: List[List[str]] = []
    for _, row in df[cols].iterrows():
        bag: List[str] = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                continue
            bag.extend(parse_descriptor_cell(v))
        # Dedupe preserving order
        seen, merged = set(), []
        for x in bag:
            if x not in seen:
                seen.add(x)
                merged.append(x)
        out.append(merged)
    return pd.Series(out, index=df.index)

def _first_valid(series: pd.Series) -> Optional[str]:
    """Return first non-empty string value in a series."""
    for v in series:
        if pd.notna(v) and str(v).strip():
            return str(v)
    return None

def _unique_sorted_nonempty(series: pd.Series) -> list[str]:
    """Unique, non-empty string values sorted alphabetically."""
    return sorted({str(x) for x in series if pd.notna(x) and str(x).strip()})

def _flatten_to_strings(obj) -> list[str]:
    """
    Recursively flatten lists/tuples/sets into a list of strings.
    Skips NAs/empties and strips whitespace.
    """
    out: list[str] = []
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple, set)):
            stack.extend(reversed(list(item)))
        elif pd.isna(item) if isinstance(item, float) else False:
            continue
        else:
            s = str(item).strip()
            if s:
                out.append(s)
    return out

def _normalize_desc_list(lst) -> list[str]:
    """
    Flatten -> lowercase -> dedupe (order-preserving).
    Uses a set internally to remove duplicates, returns a list for CSV/JSON.
    """
    flat = _flatten_to_strings(lst)
    seen: set[str] = set()
    out: list[str] = []
    for x in flat:
        s = x.lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _merge_descriptor_lists(series_of_lists: pd.Series) -> list[str]:
    """
    Merge many descriptor lists into one:
    flatten everything, lowercase, and dedupe (order-preserving).
    """
    seen: set[str] = set()
    out: list[str] = []
    for lst in series_of_lists:
        for s in _normalize_desc_list(lst):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out

def _clean_desc_token(x) -> str:
    """Lowercase, strip brackets/quotes, and trim whitespace."""
    s = str(x)
    s = re.sub(r"[\[\]\'\']", '', s)  # remove square brackets and parentheses
    return s.strip().lower()


# ---------------------------------------------------------------------
# Preparation per dataset
# ---------------------------------------------------------------------
def prepare_dataset(dataset: str, badges: dict) -> pd.DataFrame:
    """
    Data preparation for a single dataset:
      1) Require both stimuli.csv and molecules.csv.
      2) Use behavior.csv unless BEHAVIOR_FILE_MAP specifies another filename
         (fallback to behavior_1.csv if needed).
      3) Skip if odor description columns are all boolean/numeric.
      4) Parse descriptor cells as lists (handling ';' ',' '/'), combine across multiple descriptor columns.
      5) Lowercase all descriptors.
    """
    stimuli = load_csv(dataset, "stimuli.csv")
    molecules = load_csv(dataset, "molecules.csv")
    if stimuli.empty or molecules.empty:
        print(f"\nPASS: {dataset} (missing stimuli.csv or molecules.csv)")
        return pd.DataFrame()

    behavior_fname = BEHAVIOR_FILE_MAP.get(dataset, "behavior.csv")
    behavior = load_csv(dataset, behavior_fname)
    if behavior.empty and behavior_fname == "behavior.csv":
        behavior = load_csv(dataset, "behavior_1.csv")
    if behavior.empty:
        print(f"\nPASS: {dataset} (missing behavior file)")
        return pd.DataFrame()

    # Extract descriptor lists
    desc_series = extract_descriptor_lists(behavior)
    if desc_series.map(lambda lst: lst == ["__ALL_NUMERIC_OR_BOOL__"]).all():
        print(f"\nPASS: {dataset} (odor description data is only boolean/numeric)")
        return pd.DataFrame()

    # Flatten, lowercase, and de-duplicate per row
    desc_series = desc_series.apply(_normalize_desc_list)

    # Find join keys (index-aware)
    behavior, beh_stim_col = find_id_column(behavior, ID_COLS)
    stimuli,  stim_stim_col = find_id_column(stimuli,  ID_COLS)
    stimuli,  stim_cid_col  = find_id_column(stimuli,  CID_COLS)
    molecules, mol_cid_col  = find_id_column(molecules, CID_COLS)
    molecules, mol_smi_col  = find_id_column(molecules, SMILES_COLS)

    if beh_stim_col == "index":
        behavior = behavior.reset_index().rename(columns={behavior.index.name or "index": "stimulus_id"})
        beh_stim_col = "stimulus_id"
    if stim_stim_col == "index":
        stimuli = stimuli.reset_index().rename(columns={stimuli.index.name or "index": "stimulus"})
        stim_stim_col = "stimulus"

    if stim_cid_col is None or mol_cid_col is None:
        print(f"\nPASS: {dataset} (CID column not found in stimuli/molecules)")
        return pd.DataFrame()

    # Align indices
    behavior = behavior.reset_index(drop=True)
    desc_series = desc_series.reset_index(drop=True)

    # Build working table
    beh = pd.DataFrame({
        "dataset": dataset,
        "stimulus_id": behavior[beh_stim_col] if beh_stim_col in behavior.columns else pd.NA,
        "descriptors": desc_series,
    })

    # Normalize and merge
    beh["stimulus_id"] = normalize_id_series(beh["stimulus_id"])
    stimuli = stimuli.copy()
    molecules = molecules.copy()
    stimuli[stim_stim_col] = normalize_id_series(stimuli[stim_stim_col])
    stimuli[stim_cid_col]  = normalize_cid_series(stimuli[stim_cid_col])
    molecules[mol_cid_col] = normalize_cid_series(molecules[mol_cid_col])
    if mol_smi_col:
        molecules[mol_smi_col] = molecules[mol_smi_col].astype(str).str.strip()

    # behavior -> stimuli (CID)
    beh = beh.merge(
        stimuli[[stim_stim_col, stim_cid_col]].drop_duplicates(),
        left_on="stimulus_id", right_on=stim_stim_col, how="left"
    )
    beh.rename(columns={stim_cid_col: "CID"}, inplace=True)
    if stim_stim_col in beh.columns and stim_stim_col != "stimulus_id":
        beh.drop(columns=[stim_stim_col], inplace=True, errors="ignore")

    # CID -> molecules (SMILES + all other metadata; dedupe on mol_cid_col to avoid explosion)
    if mol_smi_col:
        beh = beh.merge(
            molecules.drop_duplicates(subset=[mol_cid_col]),
            left_on="CID", right_on=mol_cid_col, how="left"
        )
        if mol_cid_col in beh.columns and mol_cid_col != "CID":
            beh.drop(columns=[mol_cid_col], inplace=True, errors="ignore")
        beh.rename(columns={mol_smi_col: "SMILES"}, inplace=True)
    else:
        beh["SMILES"] = pd.NA

    # Prune and tidy
    beh = beh.dropna(subset=["descriptors", "CID"])
    beh["CID"] = normalize_cid_series(beh["CID"])
    beh = beh[["dataset", "CID", "SMILES", "descriptors"]]
    beh = beh[beh["descriptors"].map(lambda x: isinstance(x, list) and len(x) > 0)]
    return beh.reset_index(drop=True)


# ---------------------------------------------------------------------
# Consolidation across datasets
# ---------------------------------------------------------------------
def consolidate_by_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate prepared rows across datasets:
      - drop null/empty SMILES
      - group by SMILES
      - descriptors: union (order-preserving)
      - CID: first non-null, non-empty
      - datasets: sorted unique list (JSON string for CSV)
    """
    if df.empty:
        return pd.DataFrame(columns=["SMILES", "CID", "descriptors", "datasets"])

    df = df.copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df.loc[df["SMILES"].isin(["", "nan", "None"]), "SMILES"] = pd.NA
    df = df.dropna(subset=["SMILES"])

    grouped = (
        df.groupby("SMILES", as_index=False)
          .agg({
              "CID": _first_valid,
              "dataset": _unique_sorted_nonempty,
              "descriptors": _merge_descriptor_lists,
          })
    )

    grouped = grouped[grouped["descriptors"].map(lambda x: isinstance(x, list) and len(x) > 0)]
    grouped["datasets"] = grouped["dataset"].map(lambda lst: json.dumps(lst, ensure_ascii=False))
    grouped = grouped.drop(columns=["dataset"])
    grouped = grouped[["SMILES", "CID", "descriptors", "datasets"]]
    return grouped.reset_index(drop=True)


# ---------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------
def prepare_all() -> pd.DataFrame:
    """Fetch inventory, select datasets, prepare each, and consolidate across SMILES."""
    inv = get_inventory()
    datasets = filter_human_odor_datasets(inv)
    print(f"\nFound {len(datasets)} human odorCharacter datasets\n")

    all_rows: list[pd.DataFrame] = []
    for dataset, badges in tqdm(datasets):
        if dataset == "weiss_2012":  # mislabeled; contains mixtures
            continue
        df = prepare_dataset(dataset, badges)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(columns=["SMILES", "CID", "descriptors", "datasets"])

    df_all = pd.concat(all_rows, ignore_index=True)
    df_final = consolidate_by_smiles(df_all)
    return df_final


if __name__ == "__main__":
    df = prepare_all()
    print("Final dataset shape:", df.shape)
    print(df.head(10))
    out_path = DATA_DIR / "labeled_molecules_from_human_odor_datasets.csv"
    df.to_csv(out_path, index=False)
    print(f"✔ Saved: {out_path}")
