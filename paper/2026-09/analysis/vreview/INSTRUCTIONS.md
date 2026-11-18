# Video review instructions (SO-101 banana pick-and-place evaluation)

You are reviewing recorded robot evaluation episodes frame by frame. Be exhaustive: open EVERY image file assigned
to you with the Read tool (overview pages AND event pages). Do not skip pages, do not infer from a subset.
Do NOT read outcomes_visual.csv, grasp_analysis.csv or REPORT.md in the analysis folder (independent judgement).

## What the images are
Directory: /home/aivlab/kkb_capstone/2026-09-paper/analysis/vsheets/<repr>/
- epXX_ovN.jpg : overview pages, one tile per 1.0 s of the episode, in time order (18 tiles/page, 3 per row).
- epXX_evN_close.jpg / epXX_evN_open.jpg : dense pages, one tile per 0.2 s from 1 s before to 2 s after a gripper
  close/open event detected from the commanded gripper signal.
- Each tile = [top camera | side camera]. Top camera looks down at the mat: the three baskets are stacked
  vertically right of centre (blue on top, yellow middle, green bottom); the robot base/rest pose is at the top edge
  of the top image. Side camera looks from the side at the baskets (green, yellow, blue from left to right) and the
  arm comes in from the right.
- Header: representation, episode, trial, planned position/orientation/distractor (plan may not match reality –
  trust the image), and the language task, which names the TARGET basket colour.
- Tile label: time, grip cmd (commanded gripper opening, 0 = closed, ~35-40 = fully open in demonstrations),
  meas (measured gripper), gap = meas - cmd. In the demonstrations, while the banana is held in the jaws gap is
  about 9-12 (jaws blocked by the banana); closing on nothing gives gap about 0-3. z = commanded gripper height in
  cm (robot frame; the demonstrated grasp height on the mat is about -1 cm, the folded rest pose is also low,
  about -1 cm, so z alone does not tell rest from grasp — use the images).
- Distractors (apple / carrot) may be on the mat in some episodes.

## What to record per episode
Write ONE JSON object per line (JSONL) to the output file given in your task, with these keys:
- repr, ep, trial, target_color (from the task text)
- banana_start: where the banana is at the start in the top image (e.g. "left of baskets, mid height, horizontal")
- distractor_seen: none / apple / carrot / both, and where
- approached_banana: yes / no / partial — did the gripper actually go over/next to the banana before the first close?
- first_close_s and first_close_where: one of at_banana (jaws around the banana), beside_banana (near, on the mat,
  missed), in_air (well above the mat), near_rest (close to the starting pose), other (describe)
- contact: did the gripper touch or push the banana at any time (yes / no / unclear) and when
- grasped: did the banana end up held in the jaws and lifted (yes / no / unclear); use images plus gap ~10
- dropped: if grasped, did it fall before reaching a basket, when and where
- retries: number of additional close attempts after the first, and whether any worked
- basket_went_to: which basket the arm moved to for placing (blue/yellow/green/none) and whether it matches target
- release_s and release_where: in target basket / wrong basket / on basket rim / on mat / never released
- final_banana: where the banana is in the last frame
- outcome: success (banana inside target basket) / success_wrong_basket / rim / fail
- failure_point (if not success): never_approached, approached_but_missed, touched_pushed_no_grasp,
  grasped_then_dropped, placed_wrong_basket, placed_outside, arm_stuck_no_motion, other (describe)
- anomalies: oscillation/jerky motion, arm hitting baskets, camera moved, freezing, etc. with times
- timeline: a compact chronological account with times (e.g. "0-3s idle; 3-6s moves down toward rest-left;
  6.0s closes in air ~10cm; 7-11s moves to blue basket; 11s opens over blue; banana never touched")
- confidence: high / medium / low, and what was hard to see (occlusion etc.)
- pages_viewed: the list of file names you actually opened for this episode

Use plain English in the JSON. Be precise about times (read them from the tile labels).
When done, reply with a short summary table of outcome and failure_point per episode.
