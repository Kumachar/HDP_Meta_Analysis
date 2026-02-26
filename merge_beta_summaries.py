#!/usr/bin/env python3
# merge_beta_summaries.py
from pathlib import Path
import argparse, re, sys
import pandas as pd

def combine_family(results_root: Path, family: str) -> pd.DataFrame | None:
    data_dir = results_root / family / "data"
    rep_csvs = sorted(data_dir.glob("rep*/beta_summary_stats.csv"))
    if not rep_csvs:
        print(f"[{family}] no rep CSVs under {data_dir}/rep*/")
        return None

    dfs = []
    for csv_path in rep_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[skip] {csv_path}: {e}")
            continue

        rep_label = csv_path.parent.name  # e.g., 'rep63'
        if "experiment" not in df.columns:
            df["experiment"] = rep_label
        if "rep" not in df.columns:
            m = re.search(r"rep(\d+)", rep_label)
            df["rep"] = int(m.group(1)) if m else None
        df.insert(0, "family", family)
        dfs.append(df)

    if not dfs:
        print(f"[{family}] nothing to combine")
        return None

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # keep first file’s column order, then append any new columns
    cols0 = dfs[0].columns.tolist()
    combined = combined[cols0 + [c for c in combined.columns if c not in cols0]]

    # sort (optional)
    sort_cols = [c for c in ["rep", "source"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    out_path = results_root / family / "beta_summary_stats.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[{family}] wrote {out_path} — shape {combined.shape} from {len(dfs)} reps")
    return combined

def main():
    p = argparse.ArgumentParser(description="Merge per-rep beta_summary_stats.csv files")
    p.add_argument("--root", default="results_sample_2025-09-12", help="results root folder")
    p.add_argument("--families", nargs="*", default=None,
                   help="families to merge (default: all subdirs under root)")
    p.add_argument("--all-out", default="beta_summary_stats_all.csv",
                   help="filename for the global merged CSV at the root")
    args = p.parse_args()

    results_root = Path(args.root).resolve()
    if not results_root.exists():
        print(f"Root not found: {results_root}", file=sys.stderr)
        sys.exit(2)

    if args.families:
        families = args.families
    else:
        families = [d.name for d in results_root.iterdir() if d.is_dir()]

    all_dfs = []
    for fam in families:
        df = combine_family(results_root, fam)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        all_combined = pd.concat(all_dfs, ignore_index=True, sort=False)
        out_all = results_root / args.all_out
        all_combined.to_csv(out_all, index=False)
        print(f"[ALL] wrote {out_all} — shape {all_combined.shape}")
    else:
        print("[ALL] nothing to write")

if __name__ == "__main__":
    main()
