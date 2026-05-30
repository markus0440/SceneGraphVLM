# Swift GRPO scripts

This folder contains Swift-specific GRPO launch scripts and reward plugins.

## Files

```text
rewards/pvsg_psg_toon_reward.py  # PVSG/PSG TOON reward plugin
rewards/ag_toon_reward.py        # Action Genome TOON reward plugin
prompts/system_prompt.txt        # GRPO system prompt
scripts/*.sh                     # colocated vLLM + GRPO runs
server_mode/*.sh                 # separate rollout-server mode
```

Reward plugin source code was copied as-is from the working Swift experiments. Only file placement and script references were cleaned up.

## Absolute paths

The launch scripts default to:

```bash
GRPO_ROOT=/workspace/grpo
DATA_ROOT=/workspace/datasets/data_playground
MODEL_ROOT=/workspace/grpo/models
```

The defaults are absolute by design, but they can still be overridden by environment variables when needed.

## Logging

Comet API keys are not stored in the repository. Use one of these modes:

```bash
COMET_API_KEY=... bash /workspace/grpo/swift/scripts/run_pvsg_grpo_qwen3_5.sh
REPORT_TO=none bash /workspace/grpo/swift/scripts/run_pvsg_grpo_qwen3_5.sh
```
