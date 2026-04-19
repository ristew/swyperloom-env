"""Release a trained swyperloom LoRA: merge → GGUF → HuggingFace Hub.

Meant to be run on the training pod right after `prime-rl` finishes.
The adapter directory is the last checkpoint that prime-rl wrote under
`outputs/`. Three steps:

  1. Merge the LoRA weights back into the base model in memory and
     save the merged model as a local HF directory.
  2. Shell out to llama.cpp's convert_hf_to_gguf.py to produce a
     quantized .gguf file.
  3. Push the .gguf (and optionally the merged HF dir) to a HuggingFace
     Hub repo.

Dependencies (install on the pod once):
  uv pip install peft transformers huggingface_hub torch

Plus llama.cpp cloned somewhere the script can find (default: ./llama.cpp).
Set HF_TOKEN in the environment so huggingface_hub can authenticate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def merge_lora(base_model: str, adapter_dir: Path, out_dir: Path) -> None:
    """Load base + LoRA, merge, save as a standalone HF model dir."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/3] merging LoRA {adapter_dir} into {base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    print(f"      → {out_dir}")


def convert_to_gguf(
    merged_dir: Path, gguf_path: Path, quant: str, llama_cpp: Path
) -> None:
    """Invoke llama.cpp's convert_hf_to_gguf.py with a quantization type."""
    script = llama_cpp / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(
            f"{script} not found. Clone llama.cpp first:\n"
            f"  git clone https://github.com/ggerganov/llama.cpp {llama_cpp}"
        )
    print(f"[2/3] converting to GGUF ({quant}) ...")
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(merged_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            quant,
        ],
        check=True,
    )
    print(f"      → {gguf_path} ({gguf_path.stat().st_size / 2**30:.2f} GB)")


def push_to_hub(
    repo_id: str, gguf_path: Path, merged_dir: Path | None, private: bool
) -> None:
    """Create (or reuse) a Hub repo and upload the gguf + optional HF dir."""
    from huggingface_hub import HfApi, create_repo

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set; huggingface_hub can't authenticate")

    print(f"[3/3] pushing to HuggingFace Hub ({repo_id}, private={private}) ...")
    create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    if merged_dir is not None:
        api.upload_folder(folder_path=str(merged_dir), repo_id=repo_id, repo_type="model")
    print(f"      → https://huggingface.co/{repo_id}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, type=Path, help="LoRA checkpoint directory (e.g. outputs/step_200)")
    p.add_argument("--base", default="unsloth/Meta-Llama-3.1-8B", help="Base model HF id")
    p.add_argument("--repo-id", required=True, help="Destination HF repo, e.g. 'your-user/swyperloom-llama31-8b'")
    p.add_argument("--quant", default="q4_k_m", help="GGUF quantization (q4_k_m, q5_k_m, q8_0, f16, ...)")
    p.add_argument("--llama-cpp", type=Path, default=Path("llama.cpp"), help="Path to llama.cpp checkout")
    p.add_argument("--work-dir", type=Path, default=Path("release"), help="Scratch dir for merged model + gguf")
    p.add_argument("--private", action="store_true", help="Upload as a private HF repo")
    p.add_argument("--no-upload-merged", action="store_true", help="Upload only the .gguf, skip the merged HF dir")
    p.add_argument("--skip-push", action="store_true", help="Stop after building the gguf locally")
    p.add_argument("--keep-work-dir", action="store_true", help="Don't delete the scratch dir on success")
    args = p.parse_args()

    if not args.adapter.is_dir():
        p.error(f"--adapter not a directory: {args.adapter}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = args.work_dir / "merged"
    gguf_name = f"{args.repo_id.split('/')[-1]}.{args.quant}.gguf"
    gguf_path = args.work_dir / gguf_name

    merge_lora(args.base, args.adapter, merged_dir)
    convert_to_gguf(merged_dir, gguf_path, args.quant, args.llama_cpp)

    if args.skip_push:
        print(f"skipping push; gguf is at {gguf_path}")
        return 0

    push_to_hub(
        args.repo_id,
        gguf_path,
        None if args.no_upload_merged else merged_dir,
        args.private,
    )

    if not args.keep_work_dir:
        shutil.rmtree(args.work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
