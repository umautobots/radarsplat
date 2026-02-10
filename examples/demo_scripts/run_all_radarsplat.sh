#!/usr/bin/env bash

# set -e

# Usage: run_all_radarsplat.sh SEQUENCE_FILE DATA_DIR [VARIANT]
#   Mode is auto-detected from the sequence file:
#   - Training: lines "SCENE FRAME_START FRAME_END [CKPT_PATH]" (3 or 4 columns, e.g. seq_demo.txt)
#   - Render:   lines "SCENE FRAME_START FRAME_END CKPT_PATH" (4 columns, .pt required, e.g. seq_demo_render.txt)
#   VARIANT (optional): default, no_occ, no_sl, no_mp_modeling, no_noise_prob — passed to run_radarsplat.sh
SEQUENCE_FILE=$1
DATA_DIR=$2
VARIANT="${3:-}"

# Set and export so run_radarsplat.sh (invoked below) uses the same path; only edit here.
RADARSPLAT_ROOT="${RADARSPLAT_ROOT:-$HOME/repo/radarsplat}"
export RADARSPLAT_ROOT
RESULT_DIR="$RADARSPLAT_ROOT/batch_results"
if [[ -n "$VARIANT" ]]; then
  RESULT_DIR="$RESULT_DIR/$VARIANT"
fi

# Shared settings
INIT_NUM_PTS=20000
INIT_SCALE=0.5
SYNCED_LIDAR_MAP_NAME="synced_lidar_map_win5"
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
USE_WANDB=1  # Set to 0 if you want to disable wandb
CKPT=""      # Set to checkpoint path if needed, else leave empty
GPU=0

# Set default CUDA_VISIBLE_DEVICES if not already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=$GPU
fi

# Function to parse sequence file and populate EXPERIMENTS array
load_experiments_from_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "Error: Sequence file '$file' not found!"
    exit 1
  fi
  
  echo "Loading experiments from: $file"
  
  # Clear the array
  EXPERIMENTS=()
  
  # Read the file line by line (|| [[ -n "$line" ]] handles last line when file has no trailing newline)
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Remove leading/trailing whitespace and optional surrounding double-quotes
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')
    
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
      continue
    fi
    
    # Add to experiments array
    EXPERIMENTS+=("$line")
    echo "  Added: $line"
  done < "$file"
  
  echo "Loaded ${#EXPERIMENTS[@]} experiments"
}

# Load experiments from file
if [[ -z "$SEQUENCE_FILE" ]]; then
  echo "Error: SEQUENCE_FILE (first argument) required."
  exit 1
fi
load_experiments_from_file "$SEQUENCE_FILE"

# Auto-detect mode from format: if every line has 4 columns with 4th ending in .pt → render mode
# Error if file mixes training lines (3 cols or no .pt) and render lines (4 cols with .pt)
RENDER_MODE=0
if [[ ${#EXPERIMENTS[@]} -gt 0 ]]; then
  render_format_count=0
  for line in "${EXPERIMENTS[@]}"; do
    read -r scene frame_start frame_end ckpt <<< "$line"
    if [[ -n "$ckpt" && "$ckpt" =~ \.pt$ ]]; then
      ((render_format_count++)) || true
    fi
  done
  if [[ $render_format_count -gt 0 && $render_format_count -lt ${#EXPERIMENTS[@]} ]]; then
    echo "Error: Sequence file mixes training format (3 columns) and render format (4 columns with .pt)."
    echo "  Lines with .pt path: $render_format_count; other lines: $((${#EXPERIMENTS[@]} - render_format_count))"
    echo "  Use either all training lines (SCENE FRAME_START FRAME_END) or all render lines (SCENE FRAME_START FRAME_END /path/to/ckpt.pt)."
    exit 1
  fi
  all_render_format=1
  for line in "${EXPERIMENTS[@]}"; do
    read -r scene frame_start frame_end ckpt <<< "$line"
    if [[ -z "$scene" || -z "$frame_start" || ! "$frame_start" =~ ^[0-9]+$ ]] || \
       [[ -z "$frame_end" || ! "$frame_end" =~ ^[0-9]+$ ]] || \
       [[ -z "$ckpt" || ! "$ckpt" =~ \.pt$ ]]; then
      all_render_format=0
      break
    fi
  done
  if [[ "$all_render_format" -eq 1 ]]; then
    RENDER_MODE=1
    echo "=== RENDER MODE ==="
  else
    echo "=== TRAINING MODE ==="
  fi
  if [[ -n "$VARIANT" ]]; then
    echo "Variant: $VARIANT"
  fi
fi

# Validate render-mode format
if [[ "$RENDER_MODE" -eq 1 ]]; then
  for i in "${!EXPERIMENTS[@]}"; do
    line="${EXPERIMENTS[$i]}"
    read -r scene frame_start frame_end ckpt <<< "$line"
    err=""
    if [[ -z "$scene" ]]; then err="missing scene name (column 1)"; fi
    if [[ -z "$frame_start" || ! "$frame_start" =~ ^[0-9]+$ ]]; then err="${err:+$err; }invalid or missing frame_start (column 2, must be integer)"; fi
    if [[ -z "$frame_end" || ! "$frame_end" =~ ^[0-9]+$ ]]; then err="${err:+$err; }invalid or missing frame_end (column 3, must be integer)"; fi
    if [[ -z "$ckpt" ]]; then err="${err:+$err; }missing checkpoint path (column 4)"; fi
    if [[ -n "$ckpt" && ! "$ckpt" =~ \.pt$ ]]; then err="${err:+$err; }checkpoint path (column 4) should end with .pt"; fi
    if [[ -n "$err" ]]; then
      echo "Error: Render format check failed on line $((i+1)): $err"
      echo "  Line: $line"
      echo "  Expected format: SCENE_NAME FRAME_START FRAME_END /path/to/ckpt.pt"
      exit 1
    fi
  done
  echo "Render sequence format OK (${#EXPERIMENTS[@]} line(s))"
fi

for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

  if [[ "$RENDER_MODE" -eq 1 && -z "$CKPT_PATH" ]]; then
    echo "Error: Render mode but no checkpoint path (4th column) in sequence file."
    exit 1
  fi

  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

  SCRIPT_TO_RUN="$RADARSPLAT_ROOT/examples/demo_scripts/run_radarsplat.sh"

  CMD="bash $SCRIPT_TO_RUN"
  if [[ -n "$VARIANT" ]]; then
    CMD+=" --variant \"$VARIANT\""
  fi
  CMD+=" \
    --result_dir \"$RESULT_DIR\" \
    --data_dir \"$DATA_DIR\" \
    --scene_name \"$SCENE_NAME\" \
    --frame_selection $FRAME_START $FRAME_END \
    --init_num_pts $INIT_NUM_PTS \
    --init_scale $INIT_SCALE \
    --synced_lidar_map_name \"$SYNCED_LIDAR_MAP_NAME\" \
    --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""

  if [[ "$USE_WANDB" -eq 1 ]]; then
    CMD+=" --use_wandb"
  fi

  if [[ -n "$CKPT_PATH" ]]; then
    CMD+=" --ckpt \"$CKPT_PATH\""
  fi

  eval "$CMD"
done