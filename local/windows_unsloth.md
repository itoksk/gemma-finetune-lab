# Train locally on Windows with Unsloth / Windows でローカル学習 (Unsloth)

The Colab notebook *is* Unsloth — so the simplest "local-ish" path on Windows is
just to run the [Colab notebook](../notebooks/gemma4_e2b_finetune_colab.ipynb)
in your browser (free GPU, nothing to install). **Use that unless you have a real
reason to train on your own machine.**

Colab ノートブックの中身が Unsloth です。Windows でいちばん簡単なのは、ブラウザで
[Colab ノートブック](../notebooks/gemma4_e2b_finetune_colab.ipynb)を実行すること
です(無料 GPU・インストール不要)。**特別な理由がなければ Colab で十分です。**

## When local makes sense / ローカルが向いている場合

Only if you have an **NVIDIA GPU** (GeForce RTX, etc.) with **8 GB+ VRAM**.
Unsloth needs CUDA, so it does **not** run on a CPU-only laptop or on AMD/Intel
graphics. **NVIDIA GPU**(VRAM 8GB 以上)がある場合だけです。Unsloth は CUDA が
必要なので、CPU だけ・AMD/Intel GPU では動きません。

## Setup (WSL2) / セットアップ

Unsloth is happiest on Linux, so use **WSL2** (Windows Subsystem for Linux):

Unsloth は Linux 上で最も安定します。**WSL2** を使います:

```powershell
# 1) In PowerShell (Admin), install WSL2 + Ubuntu / 管理者 PowerShell で
wsl --install
# reboot, then open "Ubuntu" from the Start menu / 再起動して Ubuntu を開く
```

Make sure the **NVIDIA driver for WSL** is installed (search "CUDA on WSL" on
NVIDIA's site). Then inside Ubuntu:

```bash
# 2) Inside WSL Ubuntu / WSL の Ubuntu の中で
python3 -m venv .venv && source .venv/bin/activate
pip install unsloth
```

## Train / 学習

Use the **same code as the Colab notebook**, just as a Python script. The cells
in [`gemma4_e2b_finetune_colab.ipynb`](../notebooks/gemma4_e2b_finetune_colab.ipynb)
run unchanged outside Colab — copy them into a `.py` file, or open the notebook
in VS Code with the Jupyter extension and Run All.

Colab ノートブックと**まったく同じコード**を、スクリプトとして実行するだけです。
セルをそのまま `.py` にコピーするか、VS Code の Jupyter 拡張で開いて実行します。

Prepare data first (works on plain Windows too — it's pure Python):

```bash
python3 scripts/prepare_data.py data/examples/cafe_faq.csv
```

The notebook's **step 7** writes `my-gemma.gguf` + a `Modelfile`. Then:

```bash
ollama create my-gemma -f Modelfile
ollama run my-gemma "What time do you open?"
```

(Use `local\import_to_ollama.ps1` from PowerShell for the same thing.)

## Troubleshooting / 困ったとき

| Symptom / 症状 | Fix / 対処 |
| --- | --- |
| `CUDA not available` | No NVIDIA GPU detected — use the Colab notebook. |
| Install fails on native Windows | Use WSL2 (above); Unsloth targets Linux. |
| Out of VRAM | Set `load_in_4bit = True` in the load cell, or use Colab. |
| Don't have an NVIDIA GPU at all | Use the Colab notebook — it's the same Unsloth. |
