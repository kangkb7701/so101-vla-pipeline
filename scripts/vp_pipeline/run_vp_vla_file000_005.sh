#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/aivlab/kkb_capstone"
LEROBOT_PY="/home/aivlab/anaconda3/envs/lerobot/bin/python"
VP_VLA_PY="/home/aivlab/anaconda3/envs/vp_vla/bin/python"
SOURCE_ROOT="$PROJECT_ROOT/data/lerobot/so101_pick_place_fruit"
OVERLAY_ROOT="$PROJECT_ROOT/data/lerobot/so101_pick_place_fruit_vp_sam3_dual_file000_005"
DATASET_ROOT="$PROJECT_ROOT/data/lerobot/so101_pick_place_fruit_vp_vla/file000_005_ee7"
PROMPT_ROOT="$PROJECT_ROOT/data/vp_vla_visual_prompts"
CONFIG="$PROJECT_ROOT/configs/vp_vla/so101_file000_005_single_3090.yaml"
EXPECTED_EPISODES=180

cd "$PROJECT_ROOT"
completed=$(find "$OVERLAY_ROOT/meta/vp_sam3_overlay" -maxdepth 1 -name 'episode_*.json' | wc -l)
if [[ "$completed" -ne "$EXPECTED_EPISODES" ]]; then
  echo "SAM3 overlay incomplete: $completed/$EXPECTED_EPISODES episodes" >&2
  exit 1
fi

"$LEROBOT_PY" src/scripts/vp_pipeline/prepare_vp_vla_so101_dataset.py \
  --source_root "$SOURCE_ROOT" \
  --output_root "$DATASET_ROOT" \
  --file_indices 0-5 \
  --gripper_open_threshold_deg 20 \
  --overwrite

"$LEROBOT_PY" src/scripts/vp_pipeline/preprocess_vp_vla_so101_npz.py \
  --dataset_root "$DATASET_ROOT" \
  --overlay_root "$OVERLAY_ROOT" \
  --output_dir "$PROMPT_ROOT" \
  --dataset_name file000_005_ee7 \
  --episode_indices 0-179 \
  --overwrite

"$VP_VLA_PY" src/scripts/vp_pipeline/train_vp_vla_so101.py \
  --config_yaml "$CONFIG" \
  --validate_only

if [[ "${1:-}" == "--train" ]]; then
  "$VP_VLA_PY" src/scripts/vp_pipeline/train_vp_vla_so101.py \
    --config_yaml "$CONFIG" \
    "${@:2}"
else
  "$VP_VLA_PY" src/scripts/vp_pipeline/train_vp_vla_so101.py \
    --config_yaml "$CONFIG" \
    --validate_model
  echo "VP-VLA data and official checkpoint are ready. Re-run with --train to start training."
fi
