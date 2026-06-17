# Train locally on a Mac (Apple Silicon) with MLX / Mac でローカル学習 (MLX)

This is the **true on-device** path: your Mac does the training, nothing leaves
your computer. It needs an **Apple Silicon** Mac (M1/M2/M3/M4) with **16 GB+**
unified memory. Intel Macs and most Windows laptops should use the
[Colab notebook](../notebooks/gemma4_e2b_finetune_colab.ipynb) instead.

これは **完全にローカル** の方法です。学習は Mac の中だけで行われ、データは外に
出ません。**Apple Silicon**(M1/M2/M3/M4)・**16GB 以上**のメモリが必要です。
Intel Mac や多くの Windows は [Colab ノートブック](../notebooks/gemma4_e2b_finetune_colab.ipynb)
を使ってください。

> **Why mlx-vlm (not mlx-lm)?** Gemma 4 E2B is a brand-new *multimodal* model.
> The plain text trainer `mlx_lm.lora` can't load it yet, so we use **`mlx-vlm`**
> plus two tiny wrappers (`local/mlx_train.py`, `local/mlx_chat.py`) that handle
> the new model correctly. You just run the wrappers — they do the rest.
> Gemma 4 E2B は新しいマルチモーダルモデルなので、`mlx_lm` ではなく **`mlx-vlm`**
> と小さなラッパーを使います。あなたはラッパーを実行するだけです。

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
uv pip install mlx-vlm
```

(No `uv`? `python3.12 -m venv .venv && source .venv/bin/activate && pip install mlx-vlm`.)

---

## 2. Prepare your data / データを用意

```bash
python3 scripts/prepare_data.py data/examples/kobe_guide.csv \
    --system "You are a friendly Kobe tour guide."
```

This writes `out/train.jsonl` (the `{"messages": [...]}` format mlx-vlm reads).
`out/train.jsonl` ができます(mlx-vlm が読む形式)。

---

## 3. Train the LoRA adapter / LoRA を学習

```bash
python local/mlx_train.py --dataset out/train.jsonl --iters 300
```

- The **first run downloads the model once (~10 GB)** to your Hugging Face cache.
  初回だけモデルを約10GBダウンロードします(以後は再利用)。
- `--iters 300` is a quick run; raise to 600-1000 for stronger results.
- Training only on the assistant's replies is on by default (`--train-on-completions`).
- Any `mlx_vlm.lora` flag works too, e.g. `--lora-rank 16 --batch-size 1`.
- On an M-series Mac this takes roughly 5-20 minutes; peak memory ~11 GB.

The adapter lands in `adapters/`. 完成したアダプターは `adapters/` に入ります。

---

## 4. Chat with it / 会話してみる

```bash
python local/mlx_chat.py "What is Kobe famous for?"
```

That is "chatting with your fine-tune locally" — done, fully on your Mac.
これで「自分の Mac だけで、育てた AI と会話」できています。

---

## 5. (Optional) Take it into Ollama / (任意) Ollama に取り込む

`mlx_chat.py` already lets you use the model. If you also want it in **Ollama**
(for the [browser app](../use/web-chat/) or [vibe-local](../use/vibe-local.md)),
the **reliable** route for this brand-new model is the **Colab notebook's GGUF
export** (it produces `my-gemma.gguf` + a `Modelfile` ready for `ollama create`).

`mlx_chat.py` だけでも使えます。さらに **Ollama**(ブラウザアプリや vibe-local 用)
で使いたい場合、この新しいモデルでは **Colab ノートブックの GGUF 書き出し** が
確実です(`my-gemma.gguf` と `Modelfile` ができます)。

> Converting a Mac/MLX adapter to GGUF for Ollama is still unreliable for Gemma 4
> E2B (the tooling is catching up). So: **train + chat on the Mac with MLX**, or
> **train on Colab** when you want the Ollama / browser-app / vibe-local route.
> MLX のアダプターを GGUF に変換する経路は Gemma 4 E2B ではまだ不安定です。

---

## Troubleshooting / 困ったとき

| Symptom / 症状 | Fix / 対処 |
| --- | --- |
| `No module named mlx_vlm` | `uv pip install mlx-vlm` inside your activated venv. |
| `Received N parameters not in model` | You ran plain `mlx_lm.lora`. Use `python local/mlx_train.py` instead (it handles this). |
| `Dataset ... doesn't exist on the Hub` | Pass a real local path, e.g. `--dataset out/train.jsonl`. The wrapper loads local files. |
| Out of memory / メモリ不足 | Close other apps; try `--batch-size 1` (default). 16 GB is the practical minimum. |
| First run is slow | It's the one-time ~10 GB model download. Later runs reuse the cache. |
| Python errors on install | You're probably on Python 3.13+. Recreate the venv with 3.12 (step 1). |
