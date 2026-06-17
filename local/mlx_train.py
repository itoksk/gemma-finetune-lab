#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlx_train.py — LoRA-train Gemma 4 E2B locally on an Apple-Silicon Mac (mlx-vlm).

Gemma 4 E2B is a brand-new multimodal model, so two small fixes are needed
before mlx-vlm's trainer can use it. This wrapper applies them, then hands every
other argument straight through to `mlx_vlm.lora`:

  Fix 1 — the E2B checkpoint ships redundant KV-shared tensors (later layers
          reuse earlier layers' key/value, so the stored copies are unused).
          MLX's strict loader rejects them, so we load NON-strict (ignore them).
  Fix 2 — let `--dataset` point at a LOCAL `train.jsonl` (Hugging Face's
          load_dataset otherwise treats the path as a Hub dataset name).

Usage (inside your venv — see local/mac_mlx.md):
  python local/mlx_train.py --dataset out/train.jsonl --iters 300
  python local/mlx_train.py --iters 600 --lora-rank 16       # any mlx_vlm.lora flag works

Defaults if you omit them: model mlx-community/gemma-4-e2b-it-bf16,
dataset out/train.jsonl, output adapters/, iters 300, --train-on-completions.
"""
import os
import sys
import runpy

import mlx.nn as nn
import datasets

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-bf16"

# --- Fix 1: load non-strict so the redundant KV-shared tensors are ignored ---
_orig_load = nn.Module.load_weights
nn.Module.load_weights = lambda self, w, strict=True: _orig_load(self, w, strict=False)

# --- Fix 2: allow a local .jsonl / .csv file for --dataset ---
_hf_load = datasets.load_dataset
def _smart_load(path, *args, **kwargs):
    split = kwargs.get("split", "train")
    if isinstance(path, str) and os.path.exists(path):
        return _hf_load("json", data_files=path, split=split)
    return _hf_load(path, *args, **kwargs)
datasets.load_dataset = _smart_load


def _has(flag, argv):
    return any(a == flag for a in argv)


def main():
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        sys.argv = ["mlx_vlm.lora", "--help"]
        runpy.run_module("mlx_vlm.lora", run_name="__main__")
        return
    # Fill in friendly defaults only when the student didn't set them.
    if not _has("--model-path", argv):
        argv += ["--model-path", DEFAULT_MODEL]
    if not _has("--dataset", argv):
        argv += ["--dataset", "out/train.jsonl"]
    if not _has("--output-path", argv):
        argv += ["--output-path", "adapters"]
    if not _has("--iters", argv) and not _has("--epochs", argv):
        argv += ["--iters", "300"]
    if not _has("--train-on-completions", argv):
        argv += ["--train-on-completions"]
    sys.argv = ["mlx_vlm.lora"] + argv
    runpy.run_module("mlx_vlm.lora", run_name="__main__")


if __name__ == "__main__":
    main()
