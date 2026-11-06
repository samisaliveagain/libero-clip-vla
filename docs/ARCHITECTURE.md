# Implemented architecture

The robot, demonstrations, simulator assets, action space, and success predicates come from the same pinned LIBERO Panda stack. No embodiment retargeting, custom physics, inverse kinematics, or real-robot deployment is introduced.

## Observations and representations

| Component | Input → output | Status |
| --- | --- | --- |
| Image preprocessing | Upright RGB float32 `[B,3,128,128]` in `[0,1]` → CLIP-normalized `[B,3,224,224]` | NON-LEARNED INFRASTRUCTURE |
| CLIP vision encoder | Image → 196 spatial tokens of width 768; CLS discarded | PRETRAINED + FROZEN |
| CLIP text encoder | Instruction → up to 77 tokens of width 512, with padding mask | PRETRAINED + FROZEN |
| Visual/text projectors | Each modality → width 256 | TRAINED BY US |
| State normalizer | Seven arm angles + two finger positions → standardized `[B,9]` | NON-LEARNED INFRASTRUCTURE; training episodes only |
| State MLP | `9 → 128 → 256`, yielding one state token | TRAINED BY US |
| Fusion | 196 image + 77 text + one state + one action-query token → four transformer layers, width 256, eight heads | TRAINED BY US |
| Action head | Query representation → `256 → 128 → 7`, tanh bounds | TRAINED BY US |
| OSC_POSE and MuJoCo | Normalized pose/gripper command → torques → next observation | NON-LEARNED INFRASTRUCTURE |

CLIP is pinned to `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`. Both encoder interfaces share `data/models/clip-vit-base-patch16`. No encoder is fine-tuned.

## Training contract

The audited source observations are post-action. Within each episode, observation row `k` is paired with action row `k+1`, for `0 <= k <= T-2`. Images are flipped vertically once into an upright convention and divided by 255 once. Live OpenGL frames receive the corresponding conversion before CLIP processing.

Each task has 50 demonstrations, split into 40 train, five validation, and five test episodes. Source HDF5 files are unchanged. Dataset manifests record hashes, exact episode membership, and the training-only nine-value normalizer. Task identity selects data but is not a policy input.

The objective is mean squared error over six normalized pose coordinates plus mean squared gripper error. AdamW/Lightning update the policy components; CLIP remains frozen. Neither physics nor the controller participates in backpropagation. The selected ten-task run uses seed 20261008; exact settings are retained in `results/training_run_config.json`.

## Execution contract

At each step, a current camera image, instruction, and robot state produce one seven-value command. The existing controller applies it and MuJoCo produces the next observation. Translation components use the verified OSC scaling of up to ±0.05 m; rotation-vector components use up to ±0.5 rad. Positive gripper closes, negative opens. These are controller inputs, not absolute joint positions or physical gripper widths.

The baseline has one external camera, one current observation, and one action per decision. It has no action chunking, diffusion head, world model, reinforcement-learning objective, or ROS integration. Simulated control is 20 Hz; this does not assert hard real-time wall-clock operation.
