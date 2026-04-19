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

## 3. Install on the pod

```bash
# inside the pod
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone <your repo with this lab inside> /workspace/simulabra
cd /workspace/simulabra/environments/swyperloom

uv venv
uv pip install -e .
uv pip install prime-rl
```

## 4. Secrets

```bash
cat > secrets.env <<EOF
PRIME_API_KEY=...   # for the Kimi K2 judge (same key as hosted)
HF_TOKEN=...        # meta-llama/Llama-3.1-8B is gated on HF
EOF
```

The TOML already references `../../secrets.env` so prime-rl will pick
this up automatically.

## 5. Smoke-run 5 steps

```bash
# edit max_steps = 5 in configs/rl/swyperloom.toml just for the smoke
uv run prime-rl @configs/rl/swyperloom.toml
```

Watch for: vLLM boots on the base model, rollouts produce non-empty
text, judge JSON parses, reward distribution has variance.

## 6. Full run

Restore `max_steps = 200` and relaunch. ~30-60 min on a 3090/4090 for a
200-step LoRA-rank-16 run at batch 128.

## 7. Merge LoRA + GGUF export

```bash
uv pip install peft transformers
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
merged = PeftModel.from_pretrained(base, "outputs/checkpoints/final").merge_and_unload()
merged.save_pretrained("swyperloom-merged")
AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B").save_pretrained("swyperloom-merged")
PY

git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py swyperloom-merged \
  --outfile swyperloom-llama31-8b.gguf --outtype q4_k_m
```

## 8. Clean up

```bash
# locally, after scp-ing the .gguf off the pod
prime pods terminate <pod-id>
```

Don't leave the pod running — GPU time bills by the minute.
