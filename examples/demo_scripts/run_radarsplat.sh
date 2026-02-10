#!/usr/bin/env bash

# set -e

# Inherited from environment when run via run_all_radarsplat.sh; otherwise default below.
RADARSPLAT_ROOT="${RADARSPLAT_ROOT:-$HOME/repo/radarsplat}"

# Defaults
USE_WANDB=0
CKPT_PATH=""
VARIANT="default"
EVAL_SET="val+all"

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
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --eval_set)
      EVAL_SET="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Variant-specific trainer args (avoids duplicating full script in run_radarsplat_no_*.sh)
case "$VARIANT" in
  default)
    MULTIPATH_WEIGHT=0.6
    SPECTRAL_LEAKAGE_ARG="--spectral-leakage"
    L1OCCLOSS_LAMBDA=10
    EXTRA_TRAINER_ARGS=""
    ;;
  no_occ)
    MULTIPATH_WEIGHT=0.6
    SPECTRAL_LEAKAGE_ARG="--spectral-leakage"
    L1OCCLOSS_LAMBDA=0
    EXTRA_TRAINER_ARGS=""
    ;;
  no_sl)
    MULTIPATH_WEIGHT=0.6
    SPECTRAL_LEAKAGE_ARG="--no-spectral-leakage"
    L1OCCLOSS_LAMBDA=10
    EXTRA_TRAINER_ARGS=""
    ;;
  no_mp_modeling)
    MULTIPATH_WEIGHT=0.0
    SPECTRAL_LEAKAGE_ARG="--spectral-leakage"
    L1OCCLOSS_LAMBDA=10
    EXTRA_TRAINER_ARGS=""
    ;;
  no_noise_prob)
    MULTIPATH_WEIGHT=0.6
    SPECTRAL_LEAKAGE_ARG="--spectral-leakage"
    L1OCCLOSS_LAMBDA=10
    EXTRA_TRAINER_ARGS="--no-use_noise_probs"
    ;;
  *)
    echo "Unknown variant: $VARIANT (use: default, no_occ, no_sl, no_mp_modeling, no_noise_prob)"
    exit 1
    ;;
esac

# Verify required args
if [[ -z "$SCENE_NAME" || -z "$FRAME_SELECTION" || -z "$INIT_NUM_PTS" || -z "$INIT_SCALE" || -z "$SYNCED_LIDAR_MAP_NAME" || -z "$RADAR_AVG_MAP_NAME" ]]; then
  echo "Missing required arguments."
  echo "Usage: $0 [--variant VARIANT] --scene_name SCENE --frame_selection START END --init_num_pts N --init_scale S --synced_lidar_map_name NAME --radar_average_map_name NAME [--use_wandb] [--ckpt PATH] --result_dir DIR --data_dir DIR"
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

cd "$RADARSPLAT_ROOT"

# Run training
run_training() {
  local OPA_NOISE_REG_LOSS_LAMBDA="${1:-${OPA_NOISE_REG_LOSS_LAMBDA:-1e3}}"
  python "$RADARSPLAT_ROOT/examples/radar_simple_trainer.py" default \
      --eval_set "$EVAL_SET" \
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
      --multipath_weight "$MULTIPATH_WEIGHT" \
      $SPECTRAL_LEAKAGE_ARG \
      --sinc_width 2 \
      --max_range 50 \
      --init_num_pts "$INIT_NUM_PTS" \
      --init_opa 0.5 \
      --init_scale "$INIT_SCALE" \
      --max_steps 2000 \
      --opa_noise_reg_loss_lambda "$OPA_NOISE_REG_LOSS_LAMBDA" \
      --l1occloss_lambda "$L1OCCLOSS_LAMBDA" \
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
      $WANDB_ARGS \
      $CKPT_ARG \
      $EXTRA_TRAINER_ARGS
}

MAX_RETRIES=5
RETRY_COUNT=0

while true; do
    LOG_FILE="tmp_log/log_attempt_${RETRY_COUNT}.txt"
    echo "[INFO] Running training attempt #$((RETRY_COUNT + 1)) (variant=$VARIANT)"
    
    run_training 2>&1 | tee "$LOG_FILE"

    if grep -q "\[Nan in loss\]" "$LOG_FILE"; then
        echo "[WARNING] '[Nan in loss]' detected. Retrying after 5s..."
        ((RETRY_COUNT++))
        if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
            echo "Exceeded maximum retries ($MAX_RETRIES). Try to run code with low regularization weight."
            ((RETRY_COUNT++))
            LOG_FILE="tmp_log/log_attempt_${RETRY_COUNT}.txt"
            run_training 1e1 2>&1 | tee "$LOG_FILE"
            break
        fi
        sleep 5
    else
        echo "[INFO] Training completed. Exiting..."
        break
    fi
done
