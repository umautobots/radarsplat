#!/usr/bin/env bash

# set -e

RESULT_DIR="./batch_results/"

# Shared settings
INIT_NUM_PTS=5000 #10000 #20000
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

# # List of experiments
# EXPERIMENTS=(
#   # Sunny 1
#   "boreas-2021-09-02-11-42 27 67"
#   "boreas-2021-09-02-11-42 330 370"
#   "boreas-2021-09-02-11-42 370 410"
#   "boreas-2021-09-02-11-42 472 512" #removed in final experiments
#   # Snow
#   "boreas-2021-01-26-11-22 217 257"
#   "boreas-2021-01-26-11-22 460 500"
#   # Rain
#   "boreas-2021-04-29-15-55 190 230"
#   "boreas-2021-04-29-15-55 230 270"
#   # Night
#   "boreas-2021-09-14-20-00 80 120"
#   "boreas-2021-09-14-20-00 410 450"
#   # Sunny 2
#   "boreas-2021-04-08-12-44 50 90"
#   "boreas-2021-04-08-12-44 160 200"
#   "boreas-2021-04-08-12-44 245 285"
#   "boreas-2021-04-08-12-44 385 425"
# )

EXPERIMENTS=(
  # "boreas-2021-01-26-11-22 217 257 /home/pckung/gsplat/batch_results/boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250511_052618/ckpts/ckpt_1999_rank0.pt"
  # "boreas-2021-01-26-11-22 460 500 /home/pckung/gsplat/batch_results/boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250511_204947/ckpts/ckpt_1999_rank0.pt"
  # "boreas-2021-04-29-15-55 190 230 /home/pckung/gsplat/batch_results/boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250511_205954/ckpts/ckpt_1999_rank0.pt"
  "boreas-2021-04-29-15-55 230 270 /home/pckung/gsplat/batch_results/boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250512_002155/ckpts/ckpt_1999_rank0.pt"
  "boreas-2021-09-14-20-00 80 120 /home/pckung/gsplat/batch_results/boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250511_210949/ckpts/ckpt_1999_rank0.pt"
  "boreas-2021-09-14-20-00 410 450 /home/pckung/gsplat/batch_results/boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250512_001103/ckpts/ckpt_1999_rank0.pt"
)

# # List of evals
# EXPERIMENTS=(
#   # Snow
#   "boreas-2021-01-26-11-22 217 257 /home/pckung/gsplat/batch_results/boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250511_052618/ckpts/ckpt_1999_rank0.pt"
#   # "boreas-2021-01-26-11-22 460 500 /home/pckung/gsplat/batch_results/boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250511_204947/ckpts/ckpt_1999_rank0.pt"
#   # # Rain
#   # "boreas-2021-04-29-15-55 190 230"
#   # "boreas-2021-04-29-15-55 230 270"
#   # # Night
#   # "boreas-2021-09-14-20-00 80 120"
#   # "boreas-2021-09-14-20-00 410 450"
# )

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

# ----------------------------
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.04381_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
EXPERIMENTS=(
  # Cloudy res: 0.04381
  # "boreas-2021-10-15-12-35 55 95"
  # "boreas-2021-10-15-12-35 365 405"
)

# List of evals
# EXPERIMENTS=(
#   "boreas-2021-09-02-11-42 27 67 ckpt/.../..."
# )

for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

  CMD="bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
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