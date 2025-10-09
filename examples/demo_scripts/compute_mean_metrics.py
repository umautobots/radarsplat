#!/usr/bin/env python3
"""
Script to compute mean of all metrics from JSON files based on sequence list.
Reads a sequence list file and computes means for matching JSON files.

Usage:
    python compute_mean_metrics.py <folder_name> <sequence_list_file>

Example:
    python compute_mean_metrics.py batch_results seq_all.txt
"""

import os
import sys
import json
import glob
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def parse_sequence_list(sequence_file: str) -> List[Tuple[str, int, int]]:
    """
    Parse sequence list file to extract sequence names and frame ranges.
    
    Args:
        sequence_file: Path to the sequence list file
        
    Returns:
        List of tuples (sequence_name, start_frame, end_frame)
        
    Raises:
        FileNotFoundError: If sequence file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(sequence_file):
        raise FileNotFoundError(f"Sequence file '{sequence_file}' does not exist")
    
    sequences = []
    
    with open(sequence_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Remove quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Parse format: "sequence_name start_frame end_frame"
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid format in {sequence_file} line {line_num}: '{line}'. Expected 'sequence_name start_frame end_frame'")
            
            sequence_name = parts[0]
            try:
                start_frame = int(parts[1])
                end_frame = int(parts[2])
            except ValueError as e:
                raise ValueError(f"Invalid frame numbers in {sequence_file} line {line_num}: '{line}'. {e}")
            
            sequences.append((sequence_name, start_frame, end_frame))
    
    return sequences


def find_matching_json_files(folder_path: str, sequences: List[Tuple[str, int, int]]) -> List[str]:
    """
    Find JSON files matching the sequence patterns in the given folder.
    For duplicate sequences, choose the one with the largest timestamp.
    
    Args:
        folder_path: Path to the folder to search
        sequences: List of (sequence_name, start_frame, end_frame) tuples
        
    Returns:
        List of matching JSON file paths
    """
    matching_files = []
    used_folders = []
    
    for sequence_name, start_frame, end_frame in sequences:
        # Pattern to match the actual file structure:
        # {sequence_name}/{sequence_name}_frame_{start}_{end}_{timestamp}/stats/val_step:1999_lidarmap:True_tau:0_5.json
        pattern = os.path.join(folder_path, sequence_name, f"{sequence_name}_frame_{start_frame}_{end_frame}_*", "stats", "val_step:1999_lidarmap:True_tau:0_5.json")
        files = glob.glob(pattern, recursive=True)
        
        if files:
            # If multiple files found, choose the one with the largest timestamp
            if len(files) > 1:
                # Extract timestamps and sort by timestamp (descending)
                file_with_timestamp = []
                for file_path in files:
                    # Extract the timestamp part from the path
                    # Path format: .../boreas-2021-04-29-15-55_frame_190_230_20250511_203059/stats/...
                    folder_part = os.path.dirname(os.path.dirname(file_path))  # Get the folder containing 'stats'
                    folder_name = os.path.basename(folder_part)
                    
                    # Extract timestamp from folder name (last part after the last underscore)
                    # Format: boreas-2021-04-29-15-55_frame_190_230_20250511_203059
                    timestamp_part = folder_name.split('_')[-1]  # Get the last part (timestamp)
                    file_with_timestamp.append((file_path, folder_name, timestamp_part))
                
                # Sort by timestamp (descending) to get the latest one
                file_with_timestamp.sort(key=lambda x: x[2], reverse=True)
                selected_file = file_with_timestamp[0][0]
                selected_folder = file_with_timestamp[0][1]
                
                print(f"  Multiple runs found for {sequence_name}_frame_{start_frame}_{end_frame}, using latest: {selected_folder}")
            else:
                selected_file = files[0]
                # Extract folder name
                folder_part = os.path.dirname(os.path.dirname(selected_file))
                selected_folder = os.path.basename(folder_part)
            
            matching_files.append(selected_file)
            used_folders.append(selected_folder)
        else:
            print(f"  Warning: No files found for {sequence_name}_frame_{start_frame}_{end_frame}")
    
    # Print all used folders
    if used_folders:
        print(f"\nUsed folder names:")
        for folder in used_folders:
            print(f"  - {folder}")
    
    return matching_files


def load_metrics_from_json(file_path: str) -> Dict[str, float]:
    """
    Load all metrics from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary of metric names and their values
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or contains no numeric metrics
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract numeric metrics from the JSON data
        metrics = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        
        if not metrics:
            raise ValueError(f"No numeric metrics found in {file_path}")
        
        return metrics
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid data in {file_path}: {e}")


def compute_mean_metrics(folder_name: str, sequence_file: str) -> Dict[str, float]:
    """
    Compute mean of all metrics from JSON files matching sequences in the list file.
    
    Args:
        folder_name: Name of the folder to search
        sequence_file: Path to the sequence list file
        
    Returns:
        Dictionary with mean values for each metric
        
    Raises:
        FileNotFoundError: If folder doesn't exist or no matching files found
        ValueError: If unable to load metrics from any file
    """
    # Check if folder exists
    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"Folder '{folder_name}' does not exist")
    
    if not os.path.isdir(folder_name):
        raise ValueError(f"'{folder_name}' is not a directory")
    
    # Parse sequence list
    sequences = parse_sequence_list(sequence_file)
    print(f"Parsed {len(sequences)} sequences from {sequence_file}")
    
    # Find matching files
    matching_files = find_matching_json_files(folder_name, sequences)
    
    if not matching_files:
        raise FileNotFoundError(f"No matching JSON files found in '{folder_name}' for the sequences in '{sequence_file}'")
    
    print(f"Found {len(matching_files)} matching files:")
    for file_path in matching_files:
        print(f"  - {file_path}")
    
    # Load metrics from all files
    all_metrics = defaultdict(list)  # metric_name -> [values]
    failed_files = []
    
    for file_path in matching_files:
        try:
            metrics = load_metrics_from_json(file_path)
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
            print(f"  Loaded {len(metrics)} metrics from {os.path.basename(file_path)}")
        except Exception as e:
            failed_files.append((file_path, str(e)))
            print(f"  Warning: Failed to load {os.path.basename(file_path)}: {e}")
    
    if not all_metrics:
        raise ValueError("No valid metrics could be loaded from any file")
    
    if failed_files:
        print(f"\nWarning: Failed to load {len(failed_files)} files:")
        for file_path, error in failed_files:
            print(f"  - {os.path.basename(file_path)}: {error}")
    
    # Compute mean for each metric
    mean_metrics = {}
    for metric_name, values in all_metrics.items():
        mean_metrics[metric_name] = sum(values) / len(values)
    
    return mean_metrics, len(matching_files) - len(failed_files), len(matching_files)


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python compute_mean_metrics.py <folder_name> <sequence_list_file>")
        print("Example: python compute_mean_metrics.py batch_results seq_all.txt")
        sys.exit(1)
    
    folder_name = sys.argv[1]
    sequence_file = sys.argv[2]
    
    try:
        mean_metrics, loaded_files, total_files = compute_mean_metrics(folder_name, sequence_file)
        
        print(f"\nResults:")
        print(f"  Total files found: {total_files}")
        print(f"  Files loaded successfully: {loaded_files}")
        print(f"\nMean metrics:")
        
        # Sort metrics alphabetically for consistent output
        for metric_name in sorted(mean_metrics.keys()):
            print(f"  {metric_name}: {mean_metrics[metric_name]:.6f}")
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
