#!/usr/bin/env bash

# set -e

# Defaults
USE_WANDB=0
CKPT_PATH=""

# Parse input args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene_name)
      SCENE_NAME="$2"
      shift 2
      ;;
    --frame_selection)
      FRAME_SELECTION=("$2" "$3")
      shift 3
      ;;
    --init_num_pts)
      INIT_NUM_PTS="$2"
      shift 2
      ;;
    --init_scale)
      INIT_SCALE="$2"
      shift 2
      ;;
    --synced_lidar_map_name)
      SYNCED_LIDAR_MAP_NAME="$2"
      shift 2
      ;;
    --radar_average_map_name)
      RADAR_AVG_MAP_NAME="$2"
      shift 2
      ;;
    --use_wandb)
      USE_WANDB=1
      shift
      ;;
    --ckpt)
      CKPT_PATH="$2"
      shift 2
      ;;
    --result_dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Verify required args
if [[ -z "$SCENE_NAME" || -z "$FRAME_SELECTION" || -z "$INIT_NUM_PTS" || -z "$INIT_SCALE" || -z "$SYNCED_LIDAR_MAP_NAME" || -z "$RADAR_AVG_MAP_NAME" ]]; then
  echo "Missing required arguments."
  echo "Usage: $0 --scene_name SCENE --frame_selection START END --init_num_pts N --init_scale S --synced_lidar_map_name NAME --radar_average_map_name NAME [--use_wandb] [--ckpt PATH]"
  exit 1
fi

# Wandb args
if [[ "$USE_WANDB" -eq 1 ]]; then
  WANDB_ARGS="--wandb_img_every 100"
else
  WANDB_ARGS="--no-use-wandb"
fi

# Optional ckpt arg
CKPT_ARG=""
if [[ -n "$CKPT_PATH" ]]; then
  CKPT_ARG="--ckpt $CKPT_PATH"
fi

# Move back to project root
cd "$HOME/radarsplat"

# Run training # Config used to report number in the paper
run_training() {
  local OPA_NOISE_REG_LOSS_LAMBDA="${1:-${OPA_NOISE_REG_LOSS_LAMBDA:-1e3}}"
  python $HOME/radarsplat/examples/radar_simple_trainer.py default \
      --eval_set val+all \
      --save_fig \
      --use_lidar_map \
      --no-preprocess_thres \
      --data_factor 1 \
      --data_dir "$DATA_DIR" \
      --result_dir "$RESULT_DIR" \
      --seq_name "$SCENE_NAME" \
      --frame_selection "${FRAME_SELECTION[@]}" \
      --synced_lidar_map_name "$SYNCED_LIDAR_MAP_NAME" \
      --radar_average_map_name "$RADAR_AVG_MAP_NAME" \
      --test-every 5 \
      --radar_map_thres 0.10 \
      --multipath_weight 0.6 \
      --spectral-leakage \
      --sinc_width 2 \
      --max_range 50 \
      --init_num_pts "$INIT_NUM_PTS" \
      --init_opa 0.5 \
      --init_scale "$INIT_SCALE" \
      --max_steps 2000 \
      --opa_noise_reg_loss_lambda "$OPA_NOISE_REG_LOSS_LAMBDA" \
      --l1occloss_lambda 10 \
      --maxsize_lambda 100 \
      --sh_degree_interval 200 \
      --sh_degree 5 \
      --strategy.refine_start_iter 500 \
      --strategy.refine-stop-iter 0 \
      --strategy.reset_every 100000 \
      --strategy.prune-opa 0.0 \
      --strategy.refine-every 250 \
      --save-steps 1000 2000 \
      --eval_steps 2000 \
      --disable_viewer \
      --no-use_noise_probs \
      $WANDB_ARGS \
      $CKPT_ARG
}

MAX_RETRIES=5
RETRY_COUNT=0

while true; do
    LOG_FILE="tmp_log/log_attempt_${RETRY_COUNT}.txt"
    echo "[INFO] Running training attempt #$((RETRY_COUNT + 1))"
    
    # Run training and stream output directly to a file
    run_training 2>&1 | tee "$LOG_FILE"

    # Search the log file for error pattern
    if grep -q "\[Nan in loss\]" "$LOG_FILE"; then
        echo "[WARNING] '[Nan in loss]' detected. Retrying after 5s..."
        ((RETRY_COUNT++))
        if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
            echo "Exceeded maximum retries ($MAX_RETRIES). Try to run code with low regularization weight."
            
            ((RETRY_COUNT++))
            LOG_FILE="tmp_log/log_attempt_${RETRY_COUNT}.txt"
            # Run training with low reg weight
            run_training 1e1 2>&1 | tee "$LOG_FILE"

            break
        fi
        sleep 5
    else
        echo "[INFO] Training completed. Exiting..."
        break
    fi
done
