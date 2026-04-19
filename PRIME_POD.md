# Prime pod runbook

Training + local-vLLM rollouts run on a self-managed Prime compute pod.
Only the judge (Kimi K2 via Prime Inference) stays external. Same
billing pool as your existing Prime credits.

## 1. Check availability and prices

```bash
prime --plain availability list --gpu-type RTX4090 --gpu-count 1
# or H100_80GB for faster training at higher cost
```

A 24GB card (3090/4090) is enough for Llama-3.1-8B base + LoRA rank 16.

## 2. Create the pod

```bash
prime pods create
```

Interactive. Pick the GPU type, region, image (`ubuntu_22_cuda_12` is
the safe default). Prime provisions and hands back a pod id.

```bash
prime pods list
prime pods ssh <pod-id>
```

## 3. Install prime-rl on the pod

We follow the reverse-text example pattern — run prime-rl's official
install script, then install our env into its venv.

```bash
# inside the pod
git clone https://github.com/PrimeIntellect-ai/prime-rl
cd prime-rl
bash scripts/install.sh        # installs uv, syncs the workspace

# pull this lab into /workspace/
cd /workspace
git clone <your repo with this lab inside> simulabra

# install swyperloom into prime-rl's venv
cd /workspace/simulabra/environments/swyperloom
scripts/install-into-prime-rl.sh /workspace/prime-rl
```

## 4. Secrets

Export in the shell of the launcher tmux pane before running any
`uv run rl @` — `prime-rl`'s native CLI reads from the environment, not
from a TOML-referenced secrets file.

```bash
export PRIME_API_KEY=...   # Kimi K2 judge via Prime Inference
export HF_TOKEN=...         # meta-llama/Llama-3.1-8B is gated on HF
```

(If you prefer a file: `source /workspace/simulabra/environments/swyperloom/secrets.env`.)

## 5. Smoke-run 5 steps

```bash
cd /workspace/prime-rl
bash scripts/tmux.sh        # launcher + logs panes
# Launcher pane:
uv run inference --model.name meta-llama/Llama-3.1-8B &
# edit max_steps = 5 in .../configs/rl/swyperloom.toml for the smoke
uv run rl @ /workspace/simulabra/environments/swyperloom/configs/rl/swyperloom.toml
```

Watch for: vLLM boots on the base model, rollouts produce non-empty
text, judge JSON parses, reward distribution has variance.

## 6. Full run

Restore `max_steps = 200` and relaunch. ~30-60 min on a 3090/4090 for a
200-step LoRA-rank-16 run at batch 128.

## 7. Merge LoRA + GGUF export + push to HF Hub

`release.py` automates all three. Install the extras and llama.cpp
once, then one command:

```bash
uv pip install -e ".[release]"
git clone https://github.com/ggerganov/llama.cpp

export HF_TOKEN=hf_...   # needs write scope for the destination repo

uv run python release.py \
  --adapter outputs/step_200 \
  --repo-id your-user/swyperloom-llama31-8b \
  --quant q4_k_m
```

Flags worth knowing:
- `--private` — upload as a private HF repo.
- `--skip-push` — stop after building the .gguf locally, skip Hub upload.
- `--no-upload-merged` — ship only the .gguf, not the full merged HF dir.
- `--quant` — q4_k_m (default, ~4.6GB for 8B), q5_k_m, q8_0, f16, etc.

## 8. Clean up

```bash
# locally, after scp-ing the .gguf off the pod
prime pods terminate <pod-id>
```

Don't leave the pod running — GPU time bills by the minute.
