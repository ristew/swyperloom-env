"""Swyperloom story-continuation RL environment.

A rollout picks a random Common-Crawl story, truncates it to a random
prefix (4..len-10 words from the start), generates 4 sibling 12-token
continuations from the policy (Llama-3.1-8B base), and scores each
continuation 1-10 on diversity / interestingness / coherence /
creativity via Kimi K2.5 as LLM judge (through Prime Inference).

Reward per completion = (sum of four 1-10 scores) / 40, so all-tens gives
1.0 and worst-case gives 0.1. This is the maximize-reward flip of the
user's `loss = 10 / total_score` formulation.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

_STORIES_FILE = Path(__file__).parent / "cc-stories-short.txt"
_MIN_WORDS = 15  # 4-word min prefix + 10-word min tail
# Cap the prefix at ~380 words ≈ 490 tokens at the English-prose ratio
# (~1.3 tok/word), leaving room under seq_len=512 for the 12-token
# completion plus a small tokenizer slack. Tune if you bump seq_len.
_MAX_PREFIX_WORDS = 380
# Moonshot's K2.5 is reasoning-by-default: it narrates its thinking into
# `content` and exhausts any reasonable max_tokens before it ever reaches
# the JSON answer, even with response_format={"type":"json_object"}. We
# use the non-thinking K2 (0905) instead — same family, same judgment
# quality for a simple 1-10 scoring task, reliably emits pure JSON.
_DEFAULT_JUDGE_MODEL = "moonshotai/kimi-k2-0905"
_PRIME_BASE_URL = "https://api.pinference.ai/api/v1"

_JUDGE_PROMPT = """You are an uncompromising literary judge. These are 12-token continuations from a small language model — most will be mediocre. Be CRITICAL. Do not inflate to be polite.

Calibration:
- 1-2: broken, incoherent, contradicts prefix
- 3-4: grammatical but bland, cliche-ridden, or adds nothing
- 5-6: competent baseline — what a filler web-text snippet might look like
- 7-8: a specific detail or turn of phrase that actually lands
- 9-10: reserved for genuinely impressive craft; use rarely

Default assumption: a 12-token snippet from a small base model is around 4 unless it shows something specific to push it higher.

Score each continuation (A, B, C, D) on four criteria:
- diversity: how different this is from its 3 siblings (1 = near-duplicate of at least one sibling; 5 = same register, different words; 10 = genuinely distinct direction)
- interestingness: does this hold attention? (1 = bland filler like "and then"; 5 = ordinary prose; 10 = a concrete detail or turn that makes you want to read more)
- coherence: grammar + narrative fit given the prefix (1 = gibberish or contradicts prefix; 5 = grammatical but generic; 10 = couldn't have fit a different prefix)
- creativity: originality of word choice or idea (1 = pure cliche; 5 = predictable but competent; 10 = inventive diction or unexpected angle)

Output JSON only, no explanation:
{{"A":{{"diversity":int,"interestingness":int,"coherence":int,"creativity":int}},"B":{{...}},"C":{{...}},"D":{{...}}}}

Prefix:
{prefix}

Continuations:
A: {a}
B: {b}
C: {c}
D: {d}

JSON:"""

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _load_stories(path: Path) -> list[list[str]]:
    stories: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            words = line.split()
            if len(words) >= _MIN_WORDS:
                stories.append(words)
    return stories


def _build_dataset(stories: list[list[str]], seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for story_idx, words in enumerate(stories):
        upper = min(len(words) - 10, _MAX_PREFIX_WORDS)
        n_words = rng.randint(4, upper)
        prefix = " ".join(words[:n_words])
        rows.append({"prompt": prefix, "info": {"story_idx": story_idx, "n_words": n_words}})
    return Dataset.from_list(rows)


def _extract_continuation(completion) -> str:
    """Pull plain text out of a verifiers completion (str or Messages)."""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        last = completion[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content", "")
        if isinstance(content, list):
            parts = []
            for p in content:
                text = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts).strip()
        return str(content or "").strip()
    return ""


def _parse_scores(raw: str) -> dict:
    """Parse the judge's JSON object out of its raw text response.

    Tolerates plain JSON, fenced ```json blocks, and a leading/trailing prose.
    Raises ValueError if no object is found or it doesn't match the schema.
    """
    m = _JSON_FENCE.search(raw)
    blob = m.group(1) if m else None
    if blob is None:
        # fall back to the first {...} span
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"no JSON object in judge response: {raw[:200]!r}")
        blob = raw[start : end + 1]
    return json.loads(blob)


def _reward_from_scores(entry: dict) -> float:
    total = 0
    for k in ("diversity", "interestingness", "coherence", "creativity"):
        v = entry.get(k)
        if not isinstance(v, (int, float)):
            raise ValueError(f"missing/invalid score for {k}: {entry!r}")
        total += float(v)
    return total / 40.0


class SwyperloomJudgeRubric(vf.JudgeRubric):
    """LLM-as-judge that scores 4 sibling continuations in one call."""

    def __init__(self, judge_model: str = _DEFAULT_JUDGE_MODEL):
        # Pass a placeholder key so AsyncOpenAI doesn't complain at
        # construction time; the real key is refreshed from the env on
        # each judge call so that `load_environment()` works in contexts
        # without the key (dataset introspection, dry-runs).
        client = AsyncOpenAI(
            base_url=_PRIME_BASE_URL,
            api_key=os.environ.get("PRIME_API_KEY") or "placeholder",
        )
        super().__init__(
            judge_client=client,
            judge_model=judge_model,
            # 600 tokens is enough for the fixed 4-entry JSON schema
            # with the non-thinking K2. If you swap in a thinking model
            # (e.g. kimi-k2-thinking, kimi-k2.5) you'll need ~4000+.
            judge_sampling_args={"max_tokens": 600, "temperature": 0.0},
            judge_prompt=_JUDGE_PROMPT,
        )
        self.add_reward_func(self.score_siblings, weight=1.0)

    async def score_siblings(
        self,
        prompts: list,
        completions: list,
        states: list,
        **_: object,
    ) -> list[float]:
        """Group reward: one judge call per group of up-to-4 siblings.

        Named with plural `prompts/completions/states` so the verifiers
        Rubric dispatcher treats this as a GroupRewardFunc.
        """
        if not completions:
            return []
        # Refresh from env in case the key was exported after construction.
        self.judge_client.api_key = os.environ["PRIME_API_KEY"]
        prefix = _extract_prefix(prompts[0])
        # pad/truncate to exactly 4 for the fixed-schema judge prompt
        conts = [_extract_continuation(c) for c in completions[:4]]
        while len(conts) < 4:
            conts.append("")
        judge_prompt = self.judge_prompt.format(
            prefix=prefix, a=conts[0], b=conts[1], c=conts[2], d=conts[3]
        )
        try:
            raw = await self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"},
                **{k: v for k, v in self.judge_sampling_args.items() if v is not None},
            )
            msg = raw.choices[0].message
            # Thinking-style models often put the final answer in `content`
            # but stash reasoning in `reasoning_content` or `reasoning`.
            # If content is empty, fall back to the reasoning channel which
            # sometimes carries the JSON when the model gets confused.
            text = msg.content or ""
            if not text:
                text = getattr(msg, "reasoning_content", None) or getattr(
                    msg, "reasoning", None
                ) or ""
            scores = _parse_scores(text)
            rewards = [_reward_from_scores(scores[k]) for k in ("A", "B", "C", "D")]
        except Exception as e:
            finish = None
            try:
                finish = raw.choices[0].finish_reason  # type: ignore[name-defined]
            except Exception:
                pass
            self.logger.warning(
                f"judge call failed (finish_reason={finish}): {e}"
            )
            rewards = [0.0] * 4
        return rewards[: len(completions)]


def _extract_prefix(prompt) -> str:
    """Get the raw prefix text out of whatever form verifiers hands us."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        content = getattr(first, "content", None)
        if content is None and isinstance(first, dict):
            content = first.get("content", "")
        return str(content or "")
    return ""


def load_environment(
    stories_path: str | None = None,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    eval_size: int = 500,
    seed: int = 0,
    **_: object,
) -> vf.Environment:
    """Build the Swyperloom verifiers environment.

    Args:
        stories_path: override for the stories txt (one story per line).
        judge_model: Prime Inference model slug for the LLM judge.
        eval_size: how many stories from the tail go into the eval split.
        seed: deterministic word-cut RNG.
    """
    vf.ensure_keys(["PRIME_API_KEY"])
    path = Path(stories_path) if stories_path else _STORIES_FILE
    stories = _load_stories(path)
    if len(stories) <= eval_size:
        raise RuntimeError(
            f"only {len(stories)} usable stories in {path}; "
            f"eval_size={eval_size} leaves no training data"
        )
    train_stories = stories[:-eval_size]
    eval_stories = stories[-eval_size:]

    train_ds = _build_dataset(train_stories, seed=seed)
    eval_ds = _build_dataset(eval_stories, seed=seed + 1)

    rubric = SwyperloomJudgeRubric(judge_model=judge_model)

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        sampling_args={"max_tokens": 12, "temperature": 0.9},
    )
    return env


if __name__ == "__main__":
    import asyncio

    env = load_environment()

    print("=== dataset sample (5 prefixes) ===")
    train = env.get_dataset(n=5, seed=0)
    for i, row in enumerate(train):
        print(f"[{i}] info={row['info']}")
        print(f"    prefix={row['prompt']!r}")
        print()

    print("=== judge smoke test ===")
    prefix = train[0]["prompt"]
    fake_completions = [
        " walked silently through the dim-lit hall.",
        " walked quietly down the dim hall.",
        " exploded in a shower of confetti.",
        " aaaa aaa aaaaaa a a aaa aaa a a a a a",
    ]
    rubric = SwyperloomJudgeRubric()
    rewards = asyncio.run(
        rubric.score_siblings(
            prompts=[[{"role": "text", "content": prefix}]] * 4,
            completions=fake_completions,
            states=[{}] * 4,
        )
    )
    for letter, cont, r in zip("ABCD", fake_completions, rewards):
        print(f"{letter} ({r:.3f}): {cont!r}")
