# GRPO training

This directory contains the Swift/GRPO training entrypoints for SceneGraphVLM.
The scripts intentionally keep absolute `/workspace/...` paths because the project is expected to run inside the unified Docker/container image.

## Layout

```text
grpo/
  models/              # local checkpoints / SFT warm-starts; ignored by git
  outputs/             # optional local outputs; ignored by git
  swift/
    rewards/           # Swift external reward plugins
    prompts/           # system prompts for GRPO
    scripts/           # colocated vLLM + GRPO launch scripts
    server_mode/       # separate rollout-server launch scripts
```

## Colocated GRPO runs

Run these commands from anywhere inside the container:

```bash
# PVSG, hallucination-aware reward profile
COMET_API_KEY=... bash /workspace/grpo/swift/scripts/run_pvsg_grpo_qwen3_5.sh

# PVSG, R1-SGG-style reward profile
COMET_API_KEY=... bash /workspace/grpo/swift/scripts/run_pvsg_grpo_qwen3_5_r1sgg.sh

# PSG
COMET_API_KEY=... bash /workspace/grpo/swift/scripts/run_psg_grpo_qwen3_5.sh

# Action Genome
COMET_API_KEY=... bash /workspace/grpo/swift/scripts/run_ag_grpo_qwen3_5.sh
```

To disable external logging:

```bash
REPORT_TO=none bash /workspace/grpo/swift/scripts/run_pvsg_grpo_qwen3_5.sh
```

To resume training, pass the checkpoint path explicitly instead of editing the script:

```bash
RESUME_FROM_CHECKPOINT=/workspace/grpo/models/<run>/checkpoint-10000 \
COMET_API_KEY=... \
bash /workspace/grpo/swift/scripts/run_ag_grpo_qwen3_5.sh
```

## Expected input files

Create annotation files from the repository root first:

```bash
python datasets/tools/prepare_all_annotations.py --overwrite
```

The launch scripts then use these GRPO JSONL files inside the container:

```text
/workspace/datasets/data_playground/PVSG_json/pvsg_base_annot_gt_prompt/grpo/train_noprevgraph.jsonl
/workspace/datasets/data_playground/PVSG_json/pvsg_base_annot_gt_prompt/grpo/eval_noprevgraph.jsonl
/workspace/datasets/data_playground/PSG_json/grpo/train.jsonl
/workspace/datasets/data_playground/PSG_json/grpo/eval.jsonl
/workspace/datasets/data_playground/AG_json/grpo/train_noprevgraph.jsonl
/workspace/datasets/data_playground/AG_json/grpo/eval_noprevgraph.jsonl
```

The default warm-start checkpoints are expected under:

```text
/workspace/grpo/models/work_dirs/
```

These paths are kept absolute on purpose for the containerized setup.

## Reward plugins

The Swift reward plugins are:

```text
/workspace/grpo/swift/rewards/pvsg_psg_toon_reward.py
/workspace/grpo/swift/rewards/ag_toon_reward.py
```

The first plugin is shared by PVSG and PSG because their TOON reward parsing is compatible in the current setup. The AG plugin is separate.

## Server mode

Server mode separates rollout serving from training:

```bash
# Terminal 1: rollout server
bash /workspace/grpo/swift/server_mode/run_rollout_server.sh

# Terminal 2: GRPO training client
COMET_API_KEY=... bash /workspace/grpo/swift/server_mode/run_ag_train_with_server.sh
```
