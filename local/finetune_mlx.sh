#!/usr/bin/env bash
# finetune_mlx.sh — one command to LoRA-train Gemma 4 E2B locally on a Mac (MLX).
# Mac (Apple Silicon) で Gemma 4 E2B を LoRA 学習する簡単ラッパー。
#
# Usage / 使い方:
#   bash local/finetune_mlx.sh data/examples/kobe_guide.csv "You are a Kobe guide."
#
# Full walkthrough + troubleshooting: local/mac_mlx.md
set -euo pipefail

CSV="${1:-data/examples/cafe_faq.csv}"
SYSTEM="${2:-}"
MODEL="${MLX_MODEL:-mlx-community/gemma-4-e2b-it-bf16}"
ITERS="${ITERS:-300}"
DATA_DIR="out"
ADAPTER_DIR="adapters"

say() { printf '\033[36m==>\033[0m %s\n' "$1"; }

# Use the lab venv if it exists / ラボの venv があれば使う
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! command -v mlx_lm.lora >/dev/null 2>&1; then
  echo "mlx-lm not found. See local/mac_mlx.md step 1 (uv venv + 'uv pip install mlx-lm')." >&2
  echo "mlx-lm が見つかりません。local/mac_mlx.md の手順1を見てください。" >&2
  exit 1
fi

say "Preparing data from $CSV / データ準備"
if [ -n "$SYSTEM" ]; then
  python3 scripts/prepare_data.py "$CSV" --system "$SYSTEM" --out "$DATA_DIR"
else
  python3 scripts/prepare_data.py "$CSV" --out "$DATA_DIR"
fi

say "Training LoRA on $MODEL ($ITERS iters) / 学習中…"
mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --iters "$ITERS" \
  --batch-size 1 \
  --num-layers 8 \
  --mask-prompt \
  --adapter-path "$ADAPTER_DIR"

say "Quick test / かんたんテスト"
mlx_lm.generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_DIR" \
  --prompt "Hello! Who are you?" || true

cat <<EOF

Done. / 完了。 Adapter saved to: $ADAPTER_DIR
Next / 次:
  - Chat more:  mlx_lm.chat --model $MODEL --adapter-path $ADAPTER_DIR
  - Into Ollama / browser app / vibe-local: see local/mac_mlx.md step 5
EOF
