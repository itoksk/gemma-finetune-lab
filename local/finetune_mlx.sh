#!/usr/bin/env bash
# finetune_mlx.sh — one command to LoRA-train Gemma 4 E2B locally on a Mac (MLX).
# Mac (Apple Silicon) で Gemma 4 E2B を LoRA 学習する簡単ラッパー。
#
# Uses mlx-vlm via local/mlx_train.py (Gemma 4 E2B is multimodal, so plain
# mlx-lm can't load it). Full walkthrough: local/mac_mlx.md
#
# Usage / 使い方:
#   bash local/finetune_mlx.sh data/examples/kobe_guide.csv "You are a Kobe guide."
set -euo pipefail

CSV="${1:-data/examples/cafe_faq.csv}"
SYSTEM="${2:-}"
ITERS="${ITERS:-300}"

say() { printf '\033[36m==>\033[0m %s\n' "$1"; }

# Use the lab venv if it exists / ラボの venv があれば使う
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! python -c "import mlx_vlm" >/dev/null 2>&1; then
  say "Installing mlx-vlm... / mlx-vlm を入れます"
  if command -v uv >/dev/null 2>&1; then uv pip install -q -U mlx-vlm
  else pip install -q -U mlx-vlm; fi
fi

say "Preparing data from $CSV / データ準備"
if [ -n "$SYSTEM" ]; then
  python3 scripts/prepare_data.py "$CSV" --system "$SYSTEM"
else
  python3 scripts/prepare_data.py "$CSV"
fi

say "Training ($ITERS iters) — first run downloads ~10GB once / 初回は約10GB DL"
python local/mlx_train.py --dataset out/train.jsonl --iters "$ITERS" --output-path adapters

say "Quick chat test / かんたんテスト"
python local/mlx_chat.py "Hello! Who are you?" || true

cat <<EOF

Done. / 完了。 Adapter saved to: adapters/
Next / 次:
  - Chat more:  python local/mlx_chat.py "your question"
  - Into Ollama / browser app / vibe-local: see local/mac_mlx.md step 5
EOF
