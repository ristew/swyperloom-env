"""Single-GPU GRPO training for swyperloom via Unsloth + TRL.

Bypasses the verifiers Environment + prime-rl orchestrator in favor of
Unsloth's fast LoRA + vLLM colocation on one GPU. Reuses our existing
judge rubric (`SwyperloomJudgeRubric.score_siblings`) as a plain-
Python reward function.

Runs end-to-end on consumer hardware (12GB VRAM card comfortably):
  - Unsloth FastLanguageModel loads SmolLM2-360M with 4-bit option
  - Colocated vLLM at 40% of GPU for rollouts (num_generations=4)
  - TRL GRPOTrainer handles the GRPO loop on the same GPU
  - Our judge calls out to Kimi K2 via Prime Inference

Install:
  uv pip install "unsloth[cu124-torch240]" trl

Run:
  export PRIME_API_KEY=...
  uv run python train_unsloth.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

# Unsloth MUST be imported before transformers/trl/peft for its
# monkey-patches to take effect.
from unsloth import FastLanguageModel  # noqa: F401 — side-effect import

from trl import GRPOConfig, GRPOTrainer

from swyperloom import (
    _MAX_PREFIX_WORDS,
    _STORIES_FILE,
    _build_dataset,
    _load_stories,
    SwyperloomJudgeRubric,
)


def build_reward_func(rubric: SwyperloomJudgeRubric, num_generations: int):
    """Wrap async group-scored judge as a sync TRL reward function.

    TRL calls reward_func once per batch with a flat list of completions
    ordered as [p0_gen0, p0_gen1, ..., p0_genN, p1_gen0, ...]. We slice
    into groups of `num_generations`, judge each group (one Kimi call
    per group), and flatten rewards back in the expected order.
    """

    def reward_func(completions, prompts, **_: object) -> list[float]:
        rewards: list[float] = []
        for i in range(0, len(completions), num_generations):
            group_c = completions[i : i + num_generations]
            group_p = prompts[i : i + num_generations]
            group_rewards = asyncio.run(
                rubric.score_siblings(
                    prompts=group_p,
                    completions=group_c,
                    states=[{}] * len(group_c),
                )
            )
            rewards.extend(group_rewards)
        return rewards

    return reward_func


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/swyperloom-unsloth"))
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not os.environ.get("PRIME_API_KEY"):
        raise RuntimeError("PRIME_API_KEY not set — required for the Kimi K2 judge")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=512,
        load_in_4bit=False,  # 360M fits in bf16; flip to True for bigger bases
        fast_inference=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Base models have no chat template — apply the passthrough so that
    # TRL's internal prompt formatting emits the raw prefix as-is.
    with (Path(__file__).parent / "templates" / "completion.jinja").open() as f:
        tokenizer.chat_template = f.read()

    stories = _load_stories(_STORIES_FILE)
    train_stories = stories[: -args.eval_size]
    dataset = _build_dataset(train_stories, seed=args.seed)

    rubric = SwyperloomJudgeRubric()
    reward_func = build_reward_func(rubric, num_generations=args.num_generations)

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,  # one group per device
        gradient_accumulation_steps=4,  # effective batch = 16 rollouts / step
        max_steps=args.max_steps,
        max_prompt_length=_MAX_PREFIX_WORDS * 2,  # tokenizer slack
        max_completion_length=12,
        temperature=0.9,
        learning_rate=args.lr,
        use_vllm=True,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        save_steps=args.save_steps,
        logging_steps=10,
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[reward_func],
        train_dataset=dataset,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
