"""GRPO training entry point that runs on macOS (MPS/CPU).

The stock ``train_grpo.py`` depends on DeepSpeed, Ray, vLLM, and flash-attn,
none of which build on macOS. This module provides a slimmer alternative
built on top of ``trl.GRPOTrainer`` so you can fine-tune a small model
(e.g. Qwen3-0.6B) locally on an Apple Silicon Mac.

Trade-offs vs. the CUDA script:
- Single process, no distributed training.
- No vLLM rollouts; TRL uses HuggingFace ``generate`` internally.
- The reward model interface is simplified to a single callable that takes
  the ``(prompt, completion)`` pair and returns a scalar reward.
- Runs slowly on CPU/MPS; intended for debugging and small experiments,
  not for full-scale training.

Example:

    python -m openrlhf.cli.train_grpo_mac \\
        --pretrain Qwen/Qwen3-0.6B \\
        --dataset trl-lib/tldr \\
        --save_path outputs/grpo_mac \\
        --reward_mode length \\
        --n_samples_per_prompt 4 \\
        --num_episodes 1 \\
        --temperature 0.7
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str, bf16: bool) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if bf16:
        return torch.bfloat16
    return torch.float16


def _build_length_reward() -> Callable[[List[str], List[str]], List[float]]:
    """Toy reward: longer non-empty completions get higher scores, capped at 1.0.

    Useful as a sanity test that the training loop actually moves the model.
    Not a real reward signal.
    """

    def reward_fn(prompts: List[str], completions: List[str], **_) -> List[float]:
        scores = []
        for completion in completions:
            trimmed = completion.strip()
            if not trimmed:
                scores.append(-1.0)
                continue
            # Saturating reward: favors longer but not runaway-long outputs.
            scores.append(min(1.0, len(trimmed) / 512.0))
        return scores

    return reward_fn


def _build_prm_reward(
    prm_path: str, device: str, dtype: torch.dtype
) -> Callable[[List[str], List[str]], List[float]]:
    """Reward that uses a PRM (process reward model) checkpoint.

    The PRM is expected to be a sequence classifier: last-token logit ->
    probability the completion is correct. Falls back to zero reward on error.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer as _Tok

    tok = _Tok.from_pretrained(prm_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        prm_path, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    @torch.no_grad()
    def reward_fn(prompts: List[str], completions: List[str], **_) -> List[float]:
        texts = [p + c for p, c in zip(prompts, completions)]
        enc = tok(texts, padding=True, truncation=True, max_length=4096, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        if logits.shape[-1] == 1:
            return logits.squeeze(-1).float().cpu().tolist()
        probs = torch.softmax(logits, dim=-1)[:, -1]
        return probs.float().cpu().tolist()

    return reward_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training on macOS")
    parser.add_argument("--pretrain", required=True, help="HF model id or local path for the actor")
    parser.add_argument("--save_path", required=True, help="Output directory")
    parser.add_argument("--dataset", default="trl-lib/tldr", help="HF dataset id or local path")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--prompt_column", default="prompt")
    parser.add_argument("--max_samples", type=int, default=64, help="Cap on training prompts for quick iteration")

    parser.add_argument("--reward_mode", choices=["length", "prm"], default="length")
    parser.add_argument("--reward_pretrain", default=None, help="PRM checkpoint path (required if --reward_mode prm)")

    parser.add_argument("--n_samples_per_prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generate_max_len", type=int, default=256)
    parser.add_argument("--prompt_max_len", type=int, default=1024)

    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (requires Apple Silicon M2+/M3 or newer)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_prompts(args: argparse.Namespace) -> list:
    if os.path.exists(args.dataset):
        ds = load_dataset("json", data_files=args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, split=args.dataset_split)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(len(ds), args.max_samples)))
    if args.prompt_column not in ds.column_names:
        raise ValueError(
            f"Column '{args.prompt_column}' not found in dataset. "
            f"Available columns: {ds.column_names}"
        )
    # GRPOTrainer expects the prompt column to be named 'prompt'.
    if args.prompt_column != "prompt":
        ds = ds.rename_column(args.prompt_column, "prompt")
    keep = {"prompt"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    device = _pick_device()
    dtype = _pick_dtype(device, args.bf16)
    print(f"[train_grpo_mac] device={device} dtype={dtype}")

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        print(
            "ERROR: trl>=0.16 is required for GRPOTrainer. Install it with:\n"
            "    pip install 'trl>=0.16'\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)

    if args.reward_mode == "prm":
        if not args.reward_pretrain:
            raise ValueError("--reward_mode prm requires --reward_pretrain <path>")
        reward_fn = _build_prm_reward(args.reward_pretrain, device, dtype)
    else:
        reward_fn = _build_length_reward()

    dataset = _load_prompts(args)

    grpo_config = GRPOConfig(
        output_dir=args.save_path,
        num_train_epochs=args.num_episodes,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        remove_unused_columns=False,
        seed=args.seed,
        num_generations=args.n_samples_per_prompt,
        max_prompt_length=args.prompt_max_len,
        max_completion_length=args.generate_max_len,
        temperature=args.temperature,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"[train_grpo_mac] Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
