import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

import sys
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <sequence_file>")
    sys.exit(1)
sequence_filepath = sys.argv[1]
SEQUENCE_FILE_PATH = sequence_filepath

# Display DataFrame (df) with numbers rounded to two decimal places and not in scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)

RS_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

def _prefix_for_category(category: Optional[str]) -> str:
    if not category:
        return "seq"
    lowered = category.lower()
    if "snow" in lowered:
        return "snow"
    if "rain" in lowered:
        return "rain"
    if "night" in lowered:
        return "night"
    return "seq"

def parse_sequence_file(seq_path: Path) -> List[Dict[str, Union[str, int, None]]]:
    entries: List[Dict[str, Union[str, int, None]]] = []
    counters: Dict[str, int] = {}
    current_category: Optional[str] = None

    seq_path = Path(seq_path)
    with seq_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_category = line[1:].strip()
                continue
            value = line.strip().strip('"').strip("'")
            parts = value.split()
            if len(parts) != 3:
                print(f"[WARNING] Unexpected sequence line format: {raw_line.strip()}")
                continue
            dataset, start_frame, end_frame = parts
            prefix = _prefix_for_category(current_category)
            counters[prefix] = counters.get(prefix, 0) + 1
            name = f"{prefix}{counters[prefix]}"
            entries.append(
                {
                    "name": name,
                    "dataset": dataset,
                    "start": int(start_frame),
                    "end": int(end_frame),
                    "category": current_category,
                }
            )
    return entries

def _parse_timestamp(parts: List[str], date_format: str) -> Optional[datetime]:
    if len(parts) < 2:
        return None
    timestamp_str = "_".join(parts[-2:])
    try:
        return datetime.strptime(timestamp_str, date_format)
    except ValueError:
        return None

def _find_latest_experiment(root_path: str, dataset: str, start: int, end: int) -> Optional[str]:
    dataset_dir = os.path.join(root_path, dataset)
    if not os.path.isdir(dataset_dir):
        return None
    prefix_fragment = f"{dataset}_frame_{start}_{end}_"
    candidates: List[Tuple[Optional[datetime], str]] = []
    for entry in os.listdir(dataset_dir):
        full_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if not entry.startswith(prefix_fragment):
            continue
        timestamp = _parse_timestamp(entry.split("_"), RS_TIMESTAMP_FORMAT)
        candidates.append((timestamp, os.path.join(dataset, entry)))
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: item[0] or datetime.min)[1]
    return latest

def load_radarsplat_experiment_names(
    entries: List[Dict[str, Union[str, int, None]]],
    root_path: str = "batch_results",
) -> Dict[str, str]:
    experiments: Dict[str, str] = {}
    if not os.path.isdir(root_path):
        print(f"[WARNING] RadarSplat root path not found: {root_path}")
        return experiments
    missing_dataset_dirs: Set[str] = set()
    for entry in entries:
        name = entry["name"]
        dataset = entry["dataset"]
        start = entry["start"]
        end = entry["end"]
        if not isinstance(name, str) or not isinstance(dataset, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        dataset_dir = os.path.join(root_path, dataset)
        if not os.path.isdir(dataset_dir):
            if dataset not in missing_dataset_dirs:
                print(f"[WARNING] RadarSplat dataset directory not found: {dataset_dir}")
                missing_dataset_dirs.add(dataset)
            continue
        latest = _find_latest_experiment(root_path, dataset, start, end)
        if latest:
            experiments[name] = latest
        else:
            print(f"[WARNING] No RadarSplat experiment found for {dataset} frames {start}-{end}")
    return experiments

try:
    SEQUENCE_ENTRIES = parse_sequence_file(SEQUENCE_FILE_PATH)
except FileNotFoundError as exc:
    print(f"[WARNING] {exc}")
    SEQUENCE_ENTRIES = []

def load_radarsplat_evals_from_paths(experiment_names, root_path='batch_results', eval_file="stats/val_step:1999_lidarmap:True_tau:0_5.json"):
    rows = []
    for new_name, experiment_name in experiment_names.items():
        eval_path = os.path.join(os.path.join(root_path, experiment_name), eval_file)
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                data = json.load(f)
                data["experiment"] = new_name
        else:
            print(f"[WARNING] Missing {eval_file} in {eval_path}")

        rows.append(data)

    df = pd.DataFrame(rows)
    df = df[["experiment"] + [col for col in df.columns if col != "experiment"]]
    return df

def print_img_and_recon_eval(df, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_occ=False, show_time=False):
    img_eval_df = df[~df["experiment"].isin(img_eval_exclude_seqs)]
    img_eval_df = img_eval_df[["psnr", "ssim", "lpips"]]
    img_eval_mean_df = pd.DataFrame([img_eval_df.mean(numeric_only=True)])
    img_eval_mean_df.index = ["Image Eval. Mean"]
    print(img_eval_mean_df)
    recon_eval_df = df[~df["experiment"].isin(recon_eval_exclude_seqs)]
    # recon_eval_df_ = recon_eval_df[["RMSE", "R-CD", "accuracy"]]
    recon_eval_df_ = recon_eval_df[["RMSE", "R-CD", "accuracy", "precision", "recall"]]
    recon_eval_mean_df = pd.DataFrame([recon_eval_df_.mean(numeric_only=True)])
    recon_eval_mean_df.index = ["Recon. Eval. Mean"]
    print(recon_eval_mean_df)
    if show_occ:
        recon_eval_df_ = recon_eval_df[["Occ_RMSE", "Occ_R-CD", "Occ_accuracy"]]
        recon_eval_mean_df = pd.DataFrame([recon_eval_df_.mean(numeric_only=True)])
        recon_eval_mean_df.index = ["Recon. Eval. Mean"]
        print(recon_eval_mean_df)
    if show_time:
        recon_eval_df_ = recon_eval_df[["ellipse_time"]]
        recon_eval_mean_df = pd.DataFrame([recon_eval_df_.mean(numeric_only=True)])
        recon_eval_mean_df.index = ["Ellipse Time"]
        print(recon_eval_mean_df)

def eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_init_occ=False, eval_time=False, selected_seqs=None):
    radarsplat_experiment_names = load_radarsplat_experiment_names(SEQUENCE_ENTRIES, root_path=root_path)
    # print('Results found for evaluation:')
    # print(radarsplat_experiment_names)
    df_radarsplat = load_radarsplat_evals_from_paths(radarsplat_experiment_names, root_path=root_path)
    if selected_seqs is not None:
        print('Only use selected sequences for evaluation')
        df_radarsplat = df_radarsplat[df_radarsplat["experiment"].isin(selected_seqs)]
    print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_occ=eval_init_occ, show_time=eval_time)

if __name__ == "__main__":

    img_eval_exclude_seqs = []
    recon_eval_exclude_seqs = ["snow1", "snow2"]

    # ============================================================================
    print("------------- RadarSplat -------------")
    root_path = "batch_results"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_init_occ=True) # Also evaluate preprocessed initial occ map

    # ============================================================================
    print("------------- [Ablation] RadarSplat w/o noise probability  -------------")
    root_path = "batch_ablations/wo_noise_prob"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    print("------------- [Ablation] RadarSplat w/o multipath modeling  -------------")
    root_path = "batch_ablations/wo_mp_modeling"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    print("------------- [Ablation] RadarSplat w/o spectual leakage  -------------")
    root_path = "batch_ablations/wo_sl"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    print("------------- [Ablation] RadarSplat w/o occupancy map  -------------")
    root_path='batch_ablations/wo_occ'
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    # ============================================================================
    print("------------- [Num Gaussian Ablation] RadarSplat N=5000 -------------")
    root_path='batch_init_ablations/N_5000'
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    print("------------- [Num Gaussian Ablation] RadarSplat N=10000 -------------")
    root_path='batch_init_ablations/N_10000'
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    print("------------- [Num Gaussian Ablation] RadarSplat N=20000 -------------")
    root_path='batch_results' # default
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    print("------------- [Num Gaussian Ablation] RadarSplat N=30000 -------------")
    root_path='batch_init_ablations/N_30000'
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    # ============================================================================
    print("------------- [Size Gaussian Ablation] RadarSplat S=0.1 -------------")
    root_path='batch_init_ablations/S_0.1'
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    print("------------- [Size Gaussian Ablation] RadarSplat S=0.3 -------------")
    root_path='batch_init_ablations/S_0.3'
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    print("------------- [Size Gaussian Ablation] RadarSplat S=0.5 -------------")
    root_path='batch_results' # default
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    print("------------- [Size Gaussian Ablation] RadarSplat S=0.7 -------------")
    root_path='batch_init_ablations/S_0.7'
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, eval_time=True)

    # ============================================================================
    sunny_seqs = ["seq1", "seq2", "seq3", "seq5", "seq6", "seq7", "seq8"]
    night_seqs = ["night1", "night2"]
    rain_seqs = ["rain1", "rain2"]
    snow_seqs = ["snow1", "snow2"]

    print("------------- Sunny RadarSplat -------------")
    root_path = "batch_results"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, selected_seqs=sunny_seqs)
    
    print("------------- Night RadarSplat -------------")
    root_path = "batch_results"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, selected_seqs=night_seqs)

    print("------------- Rain RadarSplat -------------")
    root_path = "batch_results"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, selected_seqs=rain_seqs)

    print("------------- Snow RadarSplat -------------")
    root_path = "batch_results"
    eval_folder(root_path, img_eval_exclude_seqs, recon_eval_exclude_seqs, selected_seqs=snow_seqs)

