#!/usr/bin/env bash

# set -e

RESULT_DIR="./batch_results_paper/"

# Load experiments from seq_all.txt
SEQUENCE_FILE="./seq_all.txt"

# Shared settings
INIT_NUM_PTS=20000
INIT_SCALE=0.5
SYNCED_LIDAR_MAP_NAME="synced_lidar_map_win5"
# RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:30.0"
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
USE_WANDB=1  # Set to 0 if you want to disable wandb
CKPT=""      # Set to checkpoint path if needed, else leave empty
GPU=1

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
  
  # Read the file line by line
  while IFS= read -r line; do
    # Remove leading/trailing whitespace
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
      continue
    fi
    
    # Remove quotes if present
    if [[ "$line" =~ ^\".*\"$ ]]; then
      line=$(echo "$line" | sed 's/^"//;s/"$//')
    fi
    
    # Add to experiments array
    EXPERIMENTS+=("$line")
    echo "  Added: $line"
  done < "$file"
  
  echo "Loaded ${#EXPERIMENTS[@]} experiments"
}

# Load experiments from file
load_experiments_from_file "$SEQUENCE_FILE"



for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

  CMD="bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
    --result_dir $RESULT_DIR \
    --scene_name $SCENE_NAME \
    --frame_selection $FRAME_START $FRAME_END \
    --init_num_pts $INIT_NUM_PTS \
    --init_scale $INIT_SCALE \
    --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
    --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""

  if [[ "$USE_WANDB" -eq 1 ]]; then
    CMD+=" --use_wandb"
  fi

  if [[ -n "$CKPT_PATH" ]]; then
    CMD+=" --ckpt $CKPT_PATH"
  fi

  eval $CMD
done