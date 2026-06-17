#!/usr/bin/env bash
# import_to_ollama.sh — register your fine-tuned model in Ollama (macOS / Linux).
# ファインチューニングしたモデルを Ollama に登録します。
#
# Usage / 使い方:
#   bash local/import_to_ollama.sh <name> <folder-with-Modelfile>
#   e.g.  bash local/import_to_ollama.sh my-gemma ~/Downloads
#
# The <folder> must contain a `Modelfile` and the file(s) it points to
# (my-gemma.gguf, or a LoRA adapter). The Colab notebook downloads both for you.
set -euo pipefail

NAME="${1:-}"
DIR="${2:-.}"

if [ -z "$NAME" ]; then
  echo "Usage: bash local/import_to_ollama.sh <name> <folder-with-Modelfile>" >&2
  exit 2
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Get it at https://ollama.com/download" >&2
  echo "Ollama が入っていません。https://ollama.com/download から入れてください。" >&2
  exit 1
fi
if [ ! -f "$DIR/Modelfile" ]; then
  echo "No Modelfile found in: $DIR" >&2
  echo "$DIR に Modelfile がありません。" >&2
  exit 1
fi

# Make sure the Ollama server is up / サーバーが起動しているか確認
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama... / Ollama を起動中…"
  (open -a Ollama 2>/dev/null || ollama serve >/dev/null 2>&1 &) || true
  for _ in $(seq 1 30); do
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

echo "==> ollama create $NAME -f $DIR/Modelfile"
( cd "$DIR" && ollama create "$NAME" -f Modelfile )

cat <<EOF

Done! / 完了!  Try it / 試す:
  ollama run $NAME "Hello!"

Use it elsewhere / ほかでも使う:
  - Browser chat app: open use/web-chat/index.html, set the model name to "$NAME"
  - vibe-local:       see use/vibe-local.md
EOF
