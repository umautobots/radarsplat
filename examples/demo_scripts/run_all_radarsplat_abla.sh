#!/usr/bin/env bash

# set -e

SEQUENCE_FILE=$1
DATA_DIR=$2

# Set and export so run_radarsplat.sh uses the same path; only edit here (or export before calling).
RADARSPLAT_ROOT="${RADARSPLAT_ROOT:-$HOME/repo/radarsplat}"
export RADARSPLAT_ROOT

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
  
  # Read the file line by line
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')
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
load_experiments_from_file "$SEQUENCE_FILE"

#------------------------------------------------------------------------------
RESULT_DIR="./batch_ablations/wo_noise_prob/"
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"
  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>} (no_noise_prob)"
  CMD="CUDA_VISIBLE_DEVICES=$GPU bash $RADARSPLAT_ROOT/examples/demo_scripts/run_radarsplat.sh --variant no_noise_prob \
    --result_dir \"$RESULT_DIR\" \
    --data_dir \"$DATA_DIR\" \
    --scene_name \"$SCENE_NAME\" \
    --frame_selection $FRAME_START $FRAME_END \
    --init_num_pts $INIT_NUM_PTS \
    --init_scale $INIT_SCALE \
    --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
    --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""
  if [[ "$USE_WANDB" -eq 1 ]]; then CMD+=" --use_wandb"; fi
  if [[ -n "$CKPT_PATH" ]]; then CMD+=" --ckpt \"$CKPT_PATH\""; fi
  eval "$CMD"
done
#------------------------------------------------------------------------------
RESULT_DIR="./batch_ablations/wo_mp_modeling/"
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"
  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>} (no_mp_modeling)"
  CMD="CUDA_VISIBLE_DEVICES=$GPU bash $RADARSPLAT_ROOT/examples/demo_scripts/run_radarsplat.sh --variant no_mp_modeling \
    --result_dir \"$RESULT_DIR\" \
    --data_dir \"$DATA_DIR\" \
    --scene_name \"$SCENE_NAME\" \
    --frame_selection $FRAME_START $FRAME_END \
    --init_num_pts $INIT_NUM_PTS \
    --init_scale $INIT_SCALE \
    --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
    --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""
  if [[ "$USE_WANDB" -eq 1 ]]; then CMD+=" --use_wandb"; fi
  if [[ -n "$CKPT_PATH" ]]; then CMD+=" --ckpt \"$CKPT_PATH\""; fi
  eval "$CMD"
done
#------------------------------------------------------------------------------
RESULT_DIR="./batch_ablations/wo_sl/"
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"
  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>} (no_sl)"
  CMD="CUDA_VISIBLE_DEVICES=$GPU bash $RADARSPLAT_ROOT/examples/demo_scripts/run_radarsplat.sh --variant no_sl \
    --result_dir \"$RESULT_DIR\" \
    --data_dir \"$DATA_DIR\" \
    --scene_name \"$SCENE_NAME\" \
    --frame_selection $FRAME_START $FRAME_END \
    --init_num_pts $INIT_NUM_PTS \
    --init_scale $INIT_SCALE \
    --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
    --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""
  if [[ "$USE_WANDB" -eq 1 ]]; then CMD+=" --use_wandb"; fi
  if [[ -n "$CKPT_PATH" ]]; then CMD+=" --ckpt \"$CKPT_PATH\""; fi
  eval "$CMD"
done
#------------------------------------------------------------------------------
RESULT_DIR="./batch_ablations/wo_occ/"
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"
  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>} (no_occ)"
  CMD="CUDA_VISIBLE_DEVICES=$GPU bash $RADARSPLAT_ROOT/examples/demo_scripts/run_radarsplat.sh --variant no_occ \
    --result_dir \"$RESULT_DIR\" \
    --data_dir \"$DATA_DIR\" \
    --scene_name \"$SCENE_NAME\" \
    --frame_selection $FRAME_START $FRAME_END \
    --init_num_pts $INIT_NUM_PTS \
    --init_scale $INIT_SCALE \
    --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
    --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""
  if [[ "$USE_WANDB" -eq 1 ]]; then CMD+=" --use_wandb"; fi
  if [[ -n "$CKPT_PATH" ]]; then CMD+=" --ckpt \"$CKPT_PATH\""; fi
  eval "$CMD"
done
#------------------------------------------------------------------------------
RESULT_DIR="./batch_ablations/rf_occ/"
RADAR_AVG_MAP_NAME="baseline_polar_occ_filtered_polar/res:0.0596_dist:50_win_size:0_delta:10"
for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"
  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>} (rf_occ)"
  CMD="CUDA_VISIBLE_DEVICES=$GPU bash $RADARSPLAT_ROOT/examples/demo_scripts/run_radarsplat.sh \
    --result_dir \"$RESULT_DIR\" \
    --data_dir \"$DATA_DIR\" \
    --scene_name \"$SCENE_NAME\" \
    --frame_selection $FRAME_START $FRAME_END \
    --init_num_pts $INIT_NUM_PTS \
    --init_scale $INIT_SCALE \
    --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
    --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""
  if [[ "$USE_WANDB" -eq 1 ]]; then CMD+=" --use_wandb"; fi
  if [[ -n "$CKPT_PATH" ]]; then CMD+=" --ckpt \"$CKPT_PATH\""; fi
  eval "$CMD"
done