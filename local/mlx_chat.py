#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlx_chat.py — chat with your Mac-trained Gemma 4 E2B adapter (mlx-vlm).

Same non-strict-load fix as mlx_train.py, then runs mlx-vlm's generator with
your LoRA adapter applied. This is "using your fine-tune locally" — fully on
your Mac, nothing sent online.

Usage (inside your venv):
  python local/mlx_chat.py "What time do you open?"
  python local/mlx_chat.py                      # uses a default prompt
  ADAPTER=adapters MODEL=mlx-community/gemma-4-e2b-it-bf16 python local/mlx_chat.py "hi"
"""
import os
import sys
import runpy

import mlx.nn as nn

# Load non-strict (ignore Gemma 4 E2B's redundant KV-shared tensors).
_orig_load = nn.Module.load_weights
nn.Module.load_weights = lambda self, w, strict=True: _orig_load(self, w, strict=False)

MODEL = os.environ.get("MODEL", "mlx-community/gemma-4-e2b-it-bf16")
ADAPTER = os.environ.get("ADAPTER", "adapters")
prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello! Who are you?"

sys.argv = [
    "mlx_vlm.generate",
    "--model", MODEL,
    "--adapter-path", ADAPTER,
    "--prompt", prompt,
    "--max-tokens", "200",
    "--temperature", "1.0",
]
runpy.run_module("mlx_vlm.generate", run_name="__main__")
