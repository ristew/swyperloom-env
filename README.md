# swyperloom

GRPO environment for training Llama-3.1-8B base as a creative story
continuer, with Kimi K2 (via Prime Inference) as LLM judge.

## Overview

- **Environment ID**: `swyperloom`
- **Task**: complete a random-length prefix from a Common Crawl story
  with a 12-token continuation. Four siblings per prefix; judge scores
  them on diversity, interestingness, coherence, creativity (1-10 each).
- **Reward**: `(diversity + interestingness + coherence + creativity) / 40`,
  so all-tens = 1.0 and worst = 0.1. Maximize-reward flip of the
  user-spec `loss = 10 / total_score`.
- **Tags**: `creative-writing`, `completion`, `llm-judge`, `train`, `eval`

## Datasets

- **Primary dataset**: `cc-stories-short.txt` (100k Common Crawl stories,
  one per line). Shipped with this env.
- Each row has a fixed random word cut in `info.n_words`, giving per-row
  deterministic prefix length and corpus-wide variety (a 200-step run at
  batch 128 / rollouts-per-example 4 = 32 groups/step touches ~6400 rows
  out of ~99.5k train rows).
- **Eval split**: last 500 stories held out.

## Task

- **Type**: single-turn, completion mode (raw text prefix → raw text
  continuation; no chat template, no system prompt).
- **Rollouts**: 4 siblings per example via `rollouts_per_example=4` — the
  verifiers harness duplicates the row and group-scores them together,
  which is what the diversity criterion requires.

## Rubric

One group reward function, `SwyperloomJudgeRubric.score_siblings`. Makes
a single call to the judge model on Prime Inference per group of 4,
receives a JSON object with per-sibling per-criterion integer scores,
normalizes each to `[0.1, 1.0]` and returns a list of 4 floats.

## Quickstart

This dir IS the lab workspace — env code, configs, and scaffolding live
flat. Run everything from here.

```bash
cd simulabra/environments/swyperloom
echo "PRIME_API_KEY=..." >> secrets.env
echo "HF_TOKEN=..."      >> secrets.env

# install this env into the workspace venv
prime env install --path .

# sanity eval against a cheap aliased endpoint
prime eval run swyperloom -m gpt-4.1-nano -n 5 -r 4
```

The eval model (`-m`) is a *proxy* — Llama-3.1-8B base is not an API
product, so it's only available during training (prime-rl pulls the
weights from HF directly). The smoke eval just confirms rollouts
round-trip and the judge parses.

## Training (Unsloth + TRL, single GPU) — default path

One script, one GPU, no pod rental. Unsloth handles the fast LoRA
adapter + colocated vLLM rollouts; TRL's GRPOTrainer drives the GRPO
loop; our existing `SwyperloomJudgeRubric` is wrapped as a plain
Python reward function (still calling Kimi K2 via Prime Inference).

**Separate venv required** — Unsloth needs `huggingface_hub>=1.5.0`
which conflicts with verifiers' `<1` pin. One-time setup:

```bash
uv venv .venv-unsloth --python 3.12
uv pip install --python .venv-unsloth/bin/python \
  "unsloth" "trl>=0.15,<0.20" "peft>=0.10" \
  "huggingface_hub>=1.5.0,<2" "transformers>=4.48" \
  "datasets>=2.19" "openai>=1.40" "vllm" "mergekit"
```

(`trl<0.20` avoids a hard-import of `llm_blender` that has no working
PyPI release against modern transformers.)

Then train — use **plain `python`**, not `uv run`, since `uv run` will
try to reconcile `pyproject.toml`'s verifiers dep (pydantic≥2.11) with
the unsloth stack (pydantic<2.11) and fail:

```bash
source .venv-unsloth/bin/activate
export PRIME_API_KEY=...

python train_unsloth.py                       # SmolLM2-360M, 1000 steps
# or with knobs:
python train_unsloth.py --model meta-llama/Llama-3.2-1B --max-steps 500
```

After training, release with `release.py` as usual. Adapter lives at
`outputs/swyperloom-unsloth/checkpoint-<step>/`. `.venv/` stays the
verifiers-compatible env for `prime eval run`; `.venv-unsloth/` is the
training-only env.

## Training (self-managed prime-rl) — alternate path

Llama base models aren't in Prime Hosted Training's catalog, so we use
`prime-rl` self-managed — same library that backs the hosted service,
but run against a GPU you control (your machine for 1B dry-runs, a
Prime pod for the 8B run).

This follows the reverse-text example pattern from prime-rl:
https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/reverse_text/README.md

### One-time: clone prime-rl next to this repo

```bash
# wherever you keep code, e.g. ~/projects
git clone https://github.com/PrimeIntellect-ai/prime-rl
cd prime-rl
bash scripts/install.sh      # installs uv, syncs venv
```

### Wire swyperloom into that prime-rl checkout

From **this** env's dir:

```bash
scripts/install-into-prime-rl.sh ~/projects/prime-rl
```

That's a thin wrapper around `cd prime-rl && uv pip install -e /path/to/swyperloom`.

### Local dry-run (Llama-3.2-1B, your GPU)

```bash
cd ~/projects/prime-rl
bash scripts/tmux.sh                                              # opens launcher + logs panes
export PRIME_API_KEY=...  HF_TOKEN=...
uv run inference --model.name meta-llama/Llama-3.2-1B &           # vLLM on :8000
uv run rl @ ~/projects/simulabra/environments/swyperloom/configs/rl/swyperloom-1b.toml
```

### Pod run (Llama-3.1-8B, 24GB+ GPU)

See `PRIME_POD.md` for the full runbook (Prime pod rental,
secrets.env, 5-step smoke, LoRA merge, GGUF export, pod cleanup).
Same commands as above but with `swyperloom.toml` instead of
`swyperloom-1b.toml`.

## Environment arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `stories_path` | str | `cc-stories-short.txt` | Override for the stories file (one story per line, whitespace-split into words) |
| `judge_model` | str | `moonshotai/kimi-k2-0905` | Prime Inference model slug for the judge. Non-thinking K2 by default; set to `moonshotai/kimi-k2-thinking` or a larger-budget model if you want heavier reasoning at the cost of slower/more-expensive scoring. |
| `eval_size` | int | `500` | Number of stories held out at the tail for eval |
| `seed` | int | `0` | RNG seed for per-row word-cut lengths |

## Metrics

| Metric | Meaning |
| --- | --- |
| `reward` | Main scalar — `(diversity + interestingness + coherence + creativity) / 40`, range `[0.1, 1.0]` |
| `score_siblings` | Same as `reward` (only one func in the rubric) |

## Release (merge → GGUF → HF Hub)

`release.py` merges the LoRA, converts to GGUF via llama.cpp, and
pushes to HuggingFace. Run on the pod right after training completes:

```bash
uv pip install -e ".[release]"
git clone https://github.com/ggerganov/llama.cpp
export HF_TOKEN=hf_...

uv run python release.py \
  --adapter outputs/step_200 \
  --repo-id your-user/swyperloom-llama31-8b \
  --quant q4_k_m
```

See `release.py --help` for flags (`--private`, `--skip-push`,
`--quant`, `--no-upload-merged`, `--keep-work-dir`).
