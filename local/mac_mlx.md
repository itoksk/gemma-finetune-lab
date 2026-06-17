# Train locally on a Mac (Apple Silicon) with MLX / Mac でローカル学習 (MLX)

This is the **true on-device** path: your Mac does the training, nothing leaves
your computer. It needs an **Apple Silicon** Mac (M1/M2/M3/M4) with **16 GB+**
unified memory. Intel Macs and most Windows laptops should use the
[Colab notebook](../notebooks/gemma4_e2b_finetune_colab.ipynb) instead.

これは **完全にローカル** の方法です。学習は Mac の中だけで行われ、データは外に
出ません。**Apple Silicon**(M1/M2/M3/M4)・**16GB 以上**のメモリが必要です。
Intel Mac や多くの Windows は [Colab ノートブック](../notebooks/gemma4_e2b_finetune_colab.ipynb)
を使ってください。

> **New model note / 新しいモデルについて:** Gemma 4 E2B is brand new and
> multimodal. If `mlx_lm.lora` refuses to load it, see *Troubleshooting* at the
> bottom — install `mlx-vlm`, or just use the Colab notebook. Either way, the
> rest of the lab (chat, app, vibe-local) is identical.

---

## 1. One-time setup / 最初に一度だけ

MLX needs Python **3.11 or 3.12** (newer versions can break ML libraries). The
cleanest way is [`uv`](https://github.com/astral-sh/uv):

MLX には Python **3.11 / 3.12** が必要です(新しすぎると壊れることがあります)。
[`uv`](https://github.com/astral-sh/uv) を使うのが一番きれいです:

```bash
# install uv once / uv を一度だけ入れる
curl -LsSf https://astral.sh/uv/install.sh | sh

# make an isolated environment for this lab / このラボ用の隔離環境を作る
cd gemma-finetune-lab
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install mlx-lm
```

(No `uv`? `python3.12 -m venv .venv && source .venv/bin/activate && pip install mlx-lm`.)

---

## 2. Prepare your data / データを用意

```bash
python3 scripts/prepare_data.py data/examples/kobe_guide.csv \
    --system "You are a friendly Kobe tour guide."
```

This writes `out/train.jsonl` and `out/valid.jsonl` — exactly what MLX wants
(a folder with those two files). MLX は `out/` の `train.jsonl` と
`valid.jsonl` をそのまま使えます。

---

## 3. Train the LoRA adapter / LoRA を学習

```bash
mlx_lm.lora \
  --model mlx-community/gemma-4-e2b-it-bf16 \
  --train \
  --data out \
  --iters 300 \
  --batch-size 1 \
  --num-layers 8 \
  --mask-prompt \
  --adapter-path adapters
```

- `mlx-community/gemma-4-e2b-it-bf16` downloads once (~5 GB). We use **bf16**, not
  4-bit — the 4-bit Gemma 4 quants are known to be unreliable right now.
  bf16 を使います(現状 Gemma 4 の 4bit 量子化は不安定なため)。
- `--iters 300` is a quick run; raise to 600-1000 for stronger results.
- `--mask-prompt` trains only on the assistant's replies (like Colab's
  `train_on_responses_only`). AI の返答だけを学習します。
- On an M-series Mac this takes roughly 5-20 minutes depending on chip and size.

The adapter lands in `adapters/`. 完成したアダプターは `adapters/` に入ります。

---

## 4. Chat with it / 会話してみる

```bash
mlx_lm.generate \
  --model mlx-community/gemma-4-e2b-it-bf16 \
  --adapter-path adapters \
  --prompt "What is Kobe famous for?"

# or an interactive chat / 対話モード
mlx_lm.chat \
  --model mlx-community/gemma-4-e2b-it-bf16 \
  --adapter-path adapters
```

That alone is "chatting with your fine-tune locally." To use it in **Ollama**,
the browser app, or **vibe-local**, do step 5.

---

## 5. Take it into Ollama / Ollama に取り込む

Ollama gives you `ollama run`, the [browser chat app](../use/web-chat/), and
[vibe-local](../use/vibe-local.md). First fuse the adapter into the model, then
convert to GGUF with llama.cpp:

```bash
# fuse adapter + base into one model / アダプターと本体を統合
mlx_lm.fuse \
  --model mlx-community/gemma-4-e2b-it-bf16 \
  --adapter-path adapters \
  --save-path fused-model

# convert the fused model to GGUF (one-time llama.cpp setup) / GGUF に変換
git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py fused-model \
    --outfile my-gemma.gguf --outtype q8_0

# import into Ollama / Ollama に登録
printf 'FROM ./my-gemma.gguf\nPARAMETER temperature 1.0\nPARAMETER top_p 0.95\nPARAMETER top_k 64\n' > Modelfile
ollama create my-gemma -f Modelfile
ollama run my-gemma "What is Kobe famous for?"
```

Or use the helper: `bash local/import_to_ollama.sh my-gemma .`

---

## Troubleshooting / 困ったとき

| Symptom / 症状 | Fix / 対処 |
| --- | --- |
| `mlx_lm.lora` can't load Gemma 4 / 読めない | Your mlx-lm may predate Gemma 4. `uv pip install -U mlx-lm`, or try the multimodal trainer: `uv pip install mlx-vlm` then `mlx_vlm.lora --model mlx-community/gemma-4-e2b-it-bf16 --train --data out --iters 300`. Still stuck? Use the Colab notebook. |
| Out of memory / メモリ不足 | Lower `--num-layers` (e.g. 4), close other apps, or use Colab. |
| Garbled output / 出力が変 | Make sure you used the **bf16** model, not a 4-bit one. Re-fuse and re-export. |
| `convert_hf_to_gguf.py` errors on Gemma 4 | llama.cpp support for this new model may lag — use the Colab notebook's GGUF export instead, then `ollama create`. |
| Python errors on install | You're probably on Python 3.13+. Recreate the venv with 3.12 (see step 1). |
