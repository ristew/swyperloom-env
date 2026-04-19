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

## Training (self-managed on a Prime pod)

Llama-3.1-8B base is not in Prime Hosted Training's model catalog, so we
rent a raw Prime pod and run `prime-rl` ourselves. Same billing pool as
the rest of Prime.

```bash
prime pods create                 # interactive: pick 24GB+ GPU, CUDA image
prime pods ssh <pod-id>
# on the pod:
cd /workspace/simulabra/environments/swyperloom
uv pip install -e . && uv pip install prime-rl
uv run prime-rl @configs/rl/swyperloom.toml
```

See `PRIME_POD.md` for the full runbook (availability check, secrets,
5-step smoke, LoRA merge, GGUF export, pod cleanup).

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
