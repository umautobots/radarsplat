#!/usr/bin/env bash

# Usage: process_seq_paper.sh /mnt/ws-frb/projects/radar_splat/data/wave_gs/test/ boreas-2021-09-02-11-42 0.0596 

set -e  # Exit on any error

DATA_ROOT=$1 # /mnt/ws-frb/projects/radar_splat/data/boreas/
SEQ_NAME=$2
RESOLUTION=$3
# 0.0596
# 0.04381
END_INDEX=$4

if [ -z "$SEQ_NAME" ]; then
  echo "Usage: $0 <seq_name>"
  exit 1
fi

echo "Running scripts for sequence: $SEQ_NAME"

# 1. play_sync_radar_lidar.py with window_size=0
echo "Generate synced lidar"
python boreas/data_processing/play_sync_radar_lidar.py \
  --data_root "$DATA_ROOT" \
  --seq_name "$SEQ_NAME" \
  --end_index $END_INDEX \
  --save_sync_lidar \
  --window_size=0

# 2. play_sync_radar_lidar.py with window_size=5
echo "Generate synced lidar map with window size 5"
python boreas/data_processing/play_sync_radar_lidar.py \
  --data_root "$DATA_ROOT" \
  --seq_name "$SEQ_NAME" \
  --end_index $END_INDEX \
  --save_sync_lidar \
  --window_size=5

# 3. radar_grid_map.py with window_size=5
echo "Generate radar grid map with window size 5"
python boreas/data_processing/radar_grid_map.py \
  --data_root "$DATA_ROOT" \
  --seq_name "$SEQ_NAME" \
  --resolution $RESOLUTION \
  --start_index 0 \
  --end_index $END_INDEX \
  --window_size=5 \
  --save_radar_grid_map

# 4. radar_grid_map.py with radarfields and window_size=0
echo "Generate radar grid map with window size 0 and radarfields"
python boreas/data_processing/radar_grid_map.py \
  --data_root "$DATA_ROOT" \
  --seq_name "$SEQ_NAME" \
  --resolution $RESOLUTION \
  --start_index 0 \
  --end_index $END_INDEX \
  --radarfields \
  --window_size=0 \
  --save_radar_grid_map

# 5. radar_poses_saver.py
echo "Generate radar poses"
python boreas/data_processing/radar_poses_saver.py \
  --data_root "$DATA_ROOT" \
  --seq_name "$SEQ_NAME"

# 6. sync_data.py
echo "Sync data"
python boreas/data_processing/sync_data.py \
  "$DATA_ROOT"$SEQ_NAME \
  "$DATA_ROOT"synced/$SEQ_NAME

echo "Convert radar map to polar"
python boreas/data_processing/radar_map_cart2polar.py \
  --data_root "$DATA_ROOT" \
  --seq_name "$SEQ_NAME" \
  --radar_avg_map "res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0" \
  --resolution $RESOLUTION

echo "Copy sensor.yaml to the sequence folder"
cp boreas/data_processing/sensor.yaml "$DATA_ROOT"$SEQ_NAME/sensor.yaml

echo "All scripts completed for sequence: $SEQ_NAME"
