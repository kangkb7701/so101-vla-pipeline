#!/usr/bin/env bash
# Runs a fixed trial plan for one SmolVLA representation, one lerobot-rollout per trial.
# Usage: ./run_eval_session.sh <abs_joint|rel_joint|abs_ee|rel_ee> [start_trial] [plan]
#   plan = eval  (default, eval_plan/plan.csv, 20 trials)
#          eval1 (eval_plan/plan_eval1.csv, 9 trials: P1/P3/P4 x blue/yellow/green)
#   - before each trial: shows the placement, then a 7 s countdown (reset time)
#   - during a trial:  Esc = save episode and go to the next trial, Right arrow = same (end episode early),
#                      Left arrow = discard and re-record this trial; max 60 s per trial
#   - after a trial:   one-key result entry (Enter = fill in later); Ctrl+C stops the script,
#                      rerun with start_trial to continue
# Recording (one LeRobot dataset per representation and plan, one episode per trial): recordings/smolvla_<repr>/<plan>
# Results: eval_plan/results_<repr>.csv (plan eval) or eval_plan/results_<repr>_<plan>.csv
#   also stores the start pose actually used (first recorded frame: pan, wrist_roll, gripper) and the distractor
#   actually on the mat (asked after each trial; Enter = as planned)
# torch.compile on (same as the offline analysis; inference ~27 ms instead of ~90 ms). The first inference of every
#   trial compiles the model (~12 s, arm idle, nothing recorded) and counts toward episode_time_s.
# Chunk log (every predicted chunk + its noise, for exact replay): recordings/smolvla_<repr>/<plan>.chunks.csv
set -u
REPR=${1:?usage: $0 <abs_joint|rel_joint|abs_ee|rel_ee> [start_trial] [plan]}
START=${2:-1}
PLAN_NAME=${3:-eval}
case "$REPR" in abs_joint|rel_joint|abs_ee|rel_ee) ;; *) echo "unknown representation: $REPR"; exit 1;; esac
HERE=/home/aivlab/kkb_capstone/2026-09-paper/eval
if [ "$PLAN_NAME" = eval ]; then PLAN=$HERE/eval_plan/plan.csv; RESULTS=$HERE/eval_plan/results_$REPR.csv
else PLAN=$HERE/eval_plan/plan_$PLAN_NAME.csv; RESULTS=$HERE/eval_plan/results_${REPR}_$PLAN_NAME.csv; fi
[ -f "$PLAN" ] || { echo "plan not found: $PLAN"; exit 1; }
TOTAL=$(($(wc -l < "$PLAN") - 1))
CKPT=/home/aivlab/kkb_capstone/2026-09-paper/runs/smolvla_$REPR/checkpoints/012893/pretrained_model
ROOT=$HERE/recordings/smolvla_$REPR/$PLAN_NAME
PY=~/anaconda3/envs/lerobot_main/bin/python
RUN=("$PY" "$HERE/eval_rollout.py")   # = lerobot-rollout; adds IK for EE checkpoints and the chunk log
[ -f "$RESULTS" ] || echo "trial,position,orientation,distractor,color,episode_index,result,failure_type,time,distractor_actual,start_pan,start_wrist_roll,start_gripper" > "$RESULTS"

for port in 5000 5001; do
  if ss -ltn | grep -q ":$port "; then echo "camera page on port $port is running (it holds the cameras): pkill -f camera_realtime.py; pkill -f camera_align_web.py"; exit 1; fi
done

episodes() { [ -f "$ROOT/meta/info.json" ] && $PY -c "import json;print(json.load(open('$ROOT/meta/info.json'))['total_episodes'])" || echo 0; }

tail -n +2 "$PLAN" | while IFS=, read -r trial pos x y ind ori dis slot color task; do
  [ "$trial" -lt "$START" ] && continue
  echo
  echo "=================== $REPR  $PLAN_NAME  trial $trial / $TOTAL ==================="
  echo " banana: $pos (top-view pixel $x,$y) orientation $ori  $( [ "$ind" = True ] && echo '[in-dist]' || echo '[OUT-of-dist]')"
  echo " distractor: $dis ${slot:+at $slot}      target basket: $color"
  echo " map: eval_plan/position_map.jpg  |  arm at rest pose before start"
  for s in 7 6 5 4 3 2 1; do printf "\r start in %d s " $s; sleep 1; done; echo
  before=$(episodes)
  resume=(); [ "$before" -gt 0 ] && resume=(--resume=true)
  echo " (first inference compiles the model: the arm stays still for ~12 s after the 'Warmup complete' line)"
  CHUNK_LOG=$ROOT.chunks.csv EVAL_TRIAL=$trial EE_IK_LOG=$ROOT.ik.csv EE_IK_JOINT_LIMITS=${EE_IK_JOINT_LIMITS:-urdf} "${RUN[@]}" \
    --strategy.type=episodic --use_torch_compile=true \
    --policy.path=$CKPT --policy.n_action_steps=15 \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
    --robot.cameras="{camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="$task" --fps=10 \
    --dataset.repo_id=local/rollout_smolvla_${REPR}_$PLAN_NAME --dataset.root=$ROOT --dataset.single_task="$task" \
    --dataset.fps=10 --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=0 \
    --dataset.push_to_hub=false --dataset.no_stamp=true "${resume[@]}" \
    < /dev/tty 2>&1 | tee "$ROOT.trial$(printf %02d $trial).log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && echo " !! lerobot-rollout exited with an error (see the log above)"
  after=$(episodes)
  ep=$([ "$after" -gt "$before" ] && echo $((after-1)) || echo "none")
  echo " saved as episode $ep  (log: $(basename $ROOT).trial$(printf %02d $trial).log)"
  echo " result? s=success g=grasp fail d=dropped w=wrong basket c=collision/abnormal t=timeout  (Enter = later)"
  read -r -n1 res < /dev/tty; echo
  case "$res" in s) r=success; f="";; g) r=fail; f=grasp;; d) r=fail; f=dropped;; w) r=fail; f=wrong_basket;;
                  c) r=fail; f=collision;; t) r=fail; f=timeout;; *) r=""; f="";; esac
  echo " distractor actually on the mat? Enter=as planned ($dis)  n=none a=apple c=carrot b=both"
  read -r -n1 dk < /dev/tty; echo
  case "$dk" in n) da=none;; a) da=apple;; c) da=carrot;; b) da=both;; *) da=$dis;; esac
  sp=",,"
  [ "$ep" != none ] && sp=$(cd $HERE && $PY -c "
import glob, pandas as pd
d = pd.concat([pd.read_parquet(f, columns=['episode_index', 'frame_index', 'observation.state']) for f in glob.glob('$ROOT/data/**/*.parquet', recursive=True)])
s = d[(d.episode_index == $ep) & (d.frame_index == 0)]['observation.state'].iloc[0]
print(f'{s[0]:.1f},{s[4]:.1f},{s[5]:.1f}')" 2>/dev/null || echo ",,")
  echo " start pose used: pan,wrist_roll,gripper = $sp"
  echo "$trial,$pos,$ori,$dis,$color,$ep,$r,$f,$(date +%F_%T),$da,$sp" >> "$RESULTS"
done
echo "session finished: $RESULTS"
