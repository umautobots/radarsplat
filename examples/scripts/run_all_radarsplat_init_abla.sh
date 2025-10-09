#!/usr/bin/env bash

# set -e

# Shared settings
SYNCED_LIDAR_MAP_NAME="synced_lidar_map_win5"
RADAR_AVG_MAP_NAME="radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
GPU=1
USE_WANDB=1  # Set to 0 if you want to disable wandb
CKPT=""      # Set to checkpoint path if needed, else leave empty

# List of experiments
EXPERIMENTS=(
  # # Sunny 1
  # "boreas-2021-09-02-11-42 27 67"
  # "boreas-2021-09-02-11-42 330 370"
  # "boreas-2021-09-02-11-42 370 410"
  # "boreas-2021-09-02-11-42 472 512"
  # # Snow
  # "boreas-2021-01-26-11-22 217 257"
  # "boreas-2021-01-26-11-22 460 500"
  # # Rain
  # "boreas-2021-04-29-15-55 190 230"
  # "boreas-2021-04-29-15-55 230 270"
  # # Night
  # "boreas-2021-09-14-20-00 80 120"
  # "boreas-2021-09-14-20-00 410 450"
  # # Sunny 2
  # "boreas-2021-04-08-12-44 50 90"
  # "boreas-2021-04-08-12-44 160 200"
  # "boreas-2021-04-08-12-44 245 285"
  # "boreas-2021-04-08-12-44 385 425"
)

# #---------------------------------------------------------------------------
# RESULT_DIR="./batch_init_ablations/N_5000/"
# INIT_NUM_PTS=5000
# INIT_SCALE=0.5
# for EXP in "${EXPERIMENTS[@]}"; do
#   read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

#   echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

#   CMD="CUDA_VISIBLE_DEVICES=$GPU bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
#     --result_dir $RESULT_DIR \
#     --scene_name $SCENE_NAME \
#     --frame_selection $FRAME_START $FRAME_END \
#     --init_num_pts $INIT_NUM_PTS \
#     --init_scale $INIT_SCALE \
#     --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
#     --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""

#   if [[ "$USE_WANDB" -eq 1 ]]; then
#     CMD+=" --use_wandb"
#   fi

#   if [[ -n "$CKPT_PATH" ]]; then
#     CMD+=" --ckpt $CKPT_PATH"
#   fi

#   eval $CMD
# done
# #---------------------------------------------------------------------------
# RESULT_DIR="./batch_init_ablations/N_40000/"
# INIT_NUM_PTS=40000
# INIT_SCALE=0.5
# for EXP in "${EXPERIMENTS[@]}"; do
#   read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

#   echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

#   CMD="CUDA_VISIBLE_DEVICES=$GPU bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
#     --result_dir $RESULT_DIR \
#     --scene_name $SCENE_NAME \
#     --frame_selection $FRAME_START $FRAME_END \
#     --init_num_pts $INIT_NUM_PTS \
#     --init_scale $INIT_SCALE \
#     --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
#     --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""

#   if [[ "$USE_WANDB" -eq 1 ]]; then
#     CMD+=" --use_wandb"
#   fi

#   if [[ -n "$CKPT_PATH" ]]; then
#     CMD+=" --ckpt $CKPT_PATH"
#   fi

#   eval $CMD
# done
# #---------------------------------------------------------------------------
# RESULT_DIR="./batch_init_ablations/S_0.1/"
# INIT_NUM_PTS=20000
# INIT_SCALE=0.1
# for EXP in "${EXPERIMENTS[@]}"; do
#   read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

#   echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

#   CMD="CUDA_VISIBLE_DEVICES=$GPU bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
#     --result_dir $RESULT_DIR \
#     --scene_name $SCENE_NAME \
#     --frame_selection $FRAME_START $FRAME_END \
#     --init_num_pts $INIT_NUM_PTS \
#     --init_scale $INIT_SCALE \
#     --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
#     --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""

#   if [[ "$USE_WANDB" -eq 1 ]]; then
#     CMD+=" --use_wandb"
#   fi

#   if [[ -n "$CKPT_PATH" ]]; then
#     CMD+=" --ckpt $CKPT_PATH"
#   fi

#   eval $CMD
# done
# #---------------------------------------------------------------------------
# RESULT_DIR="./batch_init_ablations/S_0.3/"
# INIT_NUM_PTS=20000
# INIT_SCALE=0.3
# for EXP in "${EXPERIMENTS[@]}"; do
#   read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

#   echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

#   CMD="CUDA_VISIBLE_DEVICES=$GPU bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
#     --result_dir $RESULT_DIR \
#     --scene_name $SCENE_NAME \
#     --frame_selection $FRAME_START $FRAME_END \
#     --init_num_pts $INIT_NUM_PTS \
#     --init_scale $INIT_SCALE \
#     --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
#     --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""

#   if [[ "$USE_WANDB" -eq 1 ]]; then
#     CMD+=" --use_wandb"
#   fi

#   if [[ -n "$CKPT_PATH" ]]; then
#     CMD+=" --ckpt $CKPT_PATH"
#   fi

#   eval $CMD
# done
#---------------------------------------------------------------------------
RESULT_DIR="./batch_init_ablations/S_0.7/"
INIT_NUM_PTS=20000
INIT_SCALE=0.7
for EXP in "${EXPERIMENTS[@]}"; do
  read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

  echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

  CMD="CUDA_VISIBLE_DEVICES=$GPU bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
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
# #---------------------------------------------------------------------------
# RESULT_DIR="./batch_init_ablations/S_2.0/"
# INIT_NUM_PTS=20000
# INIT_SCALE=2.0
# for EXP in "${EXPERIMENTS[@]}"; do
#   read -r SCENE_NAME FRAME_START FRAME_END CKPT_PATH <<< "$EXP"

#   echo "Launching experiment: scene=$SCENE_NAME, frames=[$FRAME_START, $FRAME_END], ckpt=${CKPT_PATH:-<none>}"

#   CMD="CUDA_VISIBLE_DEVICES=$GPU bash $HOME/gsplat/examples/scripts/run_radarsplat.sh \
#     --result_dir $RESULT_DIR \
#     --scene_name $SCENE_NAME \
#     --frame_selection $FRAME_START $FRAME_END \
#     --init_num_pts $INIT_NUM_PTS \
#     --init_scale $INIT_SCALE \
#     --synced_lidar_map_name $SYNCED_LIDAR_MAP_NAME \
#     --radar_average_map_name \"$RADAR_AVG_MAP_NAME\""

#   if [[ "$USE_WANDB" -eq 1 ]]; then
#     CMD+=" --use_wandb"
#   fi

#   if [[ -n "$CKPT_PATH" ]]; then
#     CMD+=" --ckpt $CKPT_PATH"
#   fi

#   eval $CMD
# done