#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_notebook.py — generate notebooks/gemma4_e2b_finetune_colab.ipynb

We GENERATE the notebook from this script (don't hand-edit the .ipynb), the same
way the main course builds its Colab notebooks. Run:

    python3 scripts/build_notebook.py

The notebook is a student-data version of Unsloth's Gemma 4 (E2B) text notebook:
load YOUR data -> LoRA train -> test -> export a GGUF + LoRA adapter + Modelfile
so you can run the result locally in Ollama.
"""

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks",
                       "gemma4_e2b_finetune_colab.ipynb")

# Where students can auto-download the example data from (this repo, public).
GH_USER, GH_REPO, GH_BRANCH = "itoksk", "gemma-finetune-lab", "main"
RAW = f"https://raw.githubusercontent.com/{GH_USER}/{GH_REPO}/{GH_BRANCH}"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def build_cells():
    cells = []

    # ---------------------------------------------------------------- intro
    cells.append(md(
        "# Fine-tune your own Gemma 4 (E2B) / 自分だけの Gemma 4 (E2B) を育てる\n"
        "\n"
        "This free Colab notebook teaches a small AI to talk **your way**, using the\n"
        "examples you prepared. It runs on a **free T4 GPU** — no paid plan needed.\n"
        "\n"
        "この無料 Colab ノートブックは、あなたが用意した例を使って小さな AI に\n"
        "**あなたらしい話し方**を教えます。**無料の T4 GPU** で動きます。\n"
        "\n"
        "**Before you start / はじめる前に**\n"
        "1. Runtime ▸ Change runtime type ▸ **T4 GPU** を選択 / select **T4 GPU**.\n"
        "2. Make your data with `prepare_data.py` and keep `out/train_sharegpt.jsonl` ready.\n"
        "   `prepare_data.py` でデータを作り、`out/train_sharegpt.jsonl` を用意。\n"
        "3. Runtime ▸ **Run all** / すべて実行。\n"
        "\n"
        "Steps: **install → load Gemma 4 → add LoRA → your data → train → test → "
        "export for Ollama**.\n"
        "手順: **準備 → Gemma 4 を読込 → LoRA 追加 → データ → 学習 → テスト → "
        "Ollama 用に書き出し**。"
    ))

    # ---------------------------------------------------------------- install
    cells.append(md("## 1. Install Unsloth / Unsloth をインストール\n"
                    "Takes 1-2 minutes. / 1〜2分かかります。"))
    cells.append(code(
        "%%capture\n"
        "import os, re\n"
        'if "COLAB_" not in "".join(os.environ.keys()):\n'
        "    !pip install unsloth\n"
        "else:\n"
        "    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)\n"
        "    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, \"0.0.34\")\n"
        '    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n'
        "    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n"
        '    !pip install --no-deps --upgrade "torchao>=0.16.0"\n'
        '!pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"\n'
        "!pip install torchcodec\n"
        "import torch; torch._dynamo.config.recompile_limit = 64;"
    ))

    # ---------------------------------------------------------------- load model
    cells.append(md("## 2. Load Gemma 4 E2B / Gemma 4 E2B を読み込む\n"
                    "We use the instruct model, the same one you can run in Ollama.\n"
                    "Ollama でも動かせる instruct モデルを使います。"))
    cells.append(code(
        "from unsloth import FastModel\n"
        "import torch\n"
        "\n"
        "model, tokenizer = FastModel.from_pretrained(\n"
        '    model_name = "unsloth/gemma-4-E2B-it",\n'
        "    dtype = None,            # auto\n"
        "    max_seq_length = 1024,   # keep examples shorter than this\n"
        "    load_in_4bit = False,\n"
        "    full_finetuning = False,\n"
        ")"
    ))

    # ---------------------------------------------------------------- LoRA
    cells.append(md("## 3. Add LoRA adapters / LoRA を追加\n"
                    "LoRA trains a tiny set of extra weights (~0.2%) instead of the whole\n"
                    "model — fast, and small to share.\n"
                    "LoRA はモデル全体ではなく、ごく小さな追加の重み(約0.2%)だけを学習します。"))
    cells.append(code(
        "model = FastModel.get_peft_model(\n"
        "    model,\n"
        "    finetune_vision_layers     = False,  # text only\n"
        "    finetune_language_layers   = True,\n"
        "    finetune_attention_modules = True,\n"
        "    finetune_mlp_modules       = True,\n"
        "    r = 8, lora_alpha = 8, lora_dropout = 0,\n"
        '    bias = "none", random_state = 3407,\n'
        ")"
    ))

    # ---------------------------------------------------------------- data
    cells.append(md(
        "## 4. Load YOUR data / あなたのデータを読み込む\n"
        "Run the cell, then click **Choose Files** and pick the\n"
        "`out/train_sharegpt.jsonl` you made with `prepare_data.py`.\n"
        "セルを実行し、**ファイルを選択**から `prepare_data.py` で作った\n"
        "`out/train_sharegpt.jsonl` を選びます。\n"
        "\n"
        "*No data yet? Run the next-but-one cell to grab an example instead.*\n"
        "*まだデータが無い人は、ひとつ下の「例を使う」セルを実行してください。*"
    ))
    cells.append(code(
        "from google.colab import files\n"
        'print("Upload out/train_sharegpt.jsonl  /  out/train_sharegpt.jsonl をアップロード")\n'
        "uploaded = files.upload()\n"
        'data_file = next(iter(uploaded)) if uploaded else "train_sharegpt.jsonl"\n'
        'print("Using:", data_file)'
    ))
    cells.append(md("**Or** use an example dataset instead of uploading / "
                    "**または** アップロードせず例のデータを使う:"))
    cells.append(code(
        "# Only run this if you did NOT upload your own file above.\n"
        "# 自分のファイルをアップロードしなかった人だけ実行。\n"
        "import urllib.request\n"
        f'url = "{RAW}/data/examples/cafe_faq.csv"\n'
        '# Convert the example CSV to the format the notebook wants:\n'
        'urllib.request.urlretrieve("' + RAW + '/scripts/prepare_data.py", "prepare_data.py")\n'
        'urllib.request.urlretrieve(url, "cafe_faq.csv")\n'
        "!python3 prepare_data.py cafe_faq.csv\n"
        'data_file = "out/train_sharegpt.jsonl"\n'
        'print("Using example:", data_file)'
    ))
    cells.append(code(
        "from datasets import load_dataset\n"
        'dataset = load_dataset("json", data_files=data_file, split="train")\n'
        'print(dataset)\n'
        "print(dataset[0])"
    ))

    cells.append(md("Apply the Gemma 4 chat template and turn each conversation into\n"
                    "one training string.\n"
                    "Gemma 4 のチャットテンプレートを当て、各会話を1つの学習用テキストにします。"))
    cells.append(code(
        "from unsloth.chat_templates import get_chat_template, standardize_data_formats\n"
        'tokenizer = get_chat_template(tokenizer, chat_template = "gemma-4")\n'
        "dataset = standardize_data_formats(dataset)\n"
        "\n"
        "def formatting_prompts_func(examples):\n"
        '    convos = examples["conversations"]\n'
        "    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)"
        ".removeprefix('<bos>') for c in convos]\n"
        '    return {"text": texts}\n'
        "\n"
        "dataset = dataset.map(formatting_prompts_func, batched=True)\n"
        'print(dataset[0]["text"])'
    ))

    # ---------------------------------------------------------------- train
    cells.append(md(
        "## 5. Train / 学習する\n"
        "For a small custom dataset, training for a few **epochs** (full passes over\n"
        "your data) works better than a fixed step count. 2-4 is a good range.\n"
        "小さな自作データでは、固定ステップより数 **エポック**(データを丸ごと何周\n"
        "するか)の方が良い結果になります。2〜4 が目安です。"
    ))
    cells.append(code(
        "EPOCHS = 3   # try 2-4 / 2〜4 で試す\n"
        "\n"
        "from trl import SFTTrainer, SFTConfig\n"
        "trainer = SFTTrainer(\n"
        "    model = model, tokenizer = tokenizer,\n"
        "    train_dataset = dataset, eval_dataset = None,\n"
        "    args = SFTConfig(\n"
        '        dataset_text_field = "text",\n'
        "        per_device_train_batch_size = 1,\n"
        "        gradient_accumulation_steps = 4,\n"
        "        warmup_steps = 5,\n"
        "        num_train_epochs = EPOCHS,\n"
        "        learning_rate = 2e-4,\n"
        "        logging_steps = 1,\n"
        '        optim = "adamw_8bit",\n'
        "        weight_decay = 0.001,\n"
        '        lr_scheduler_type = "linear",\n'
        "        seed = 3407,\n"
        '        report_to = "none",\n'
        "    ),\n"
        ")"
    ))
    cells.append(md("Train only on the assistant's replies (not the user's questions) —\n"
                    "this makes the fine-tune sharper.\n"
                    "ユーザーの質問ではなく AI の返答だけを学習対象にします。精度が上がります。"))
    cells.append(code(
        "from unsloth.chat_templates import train_on_responses_only\n"
        "trainer = train_on_responses_only(\n"
        "    trainer,\n"
        '    instruction_part = "<|turn>user\\n",\n'
        '    response_part = "<|turn>model\\n",\n'
        ")"
    ))
    cells.append(code("trainer_stats = trainer.train()\n"
                      'print("Done! / 完了!", trainer_stats.metrics.get("train_runtime"), "seconds")'))

    # ---------------------------------------------------------------- test
    cells.append(md("## 6. Test it / 試す\n"
                    "Change the question and re-run to chat with your fine-tune.\n"
                    "質問を書き換えて再実行すると、育てた AI と会話できます。"))
    cells.append(code(
        "from transformers import TextStreamer\n"
        "messages = [{\n"
        '    "role": "user",\n'
        '    "content": [{"type": "text", "text": "What time do you open?"}]\n'
        "}]\n"
        "inputs = tokenizer.apply_chat_template(\n"
        "    messages, add_generation_prompt=True, return_tensors='pt',\n"
        '    tokenize=True, return_dict=True,\n'
        ').to("cuda")\n'
        "_ = model.generate(**inputs, max_new_tokens=128,\n"
        "                   temperature=1.0, top_p=0.95, top_k=64,\n"
        "                   streamer=TextStreamer(tokenizer, skip_prompt=True))"
    ))

    # ---------------------------------------------------------------- export
    cells.append(md(
        "## 7. Export for Ollama / Ollama 用に書き出す\n"
        "Two ways to take your model home — run the one you want.\n"
        "持ち帰る方法は2つ。使いたい方を実行してください。\n"
        "\n"
        "- **A. Merged GGUF (recommended / おすすめ)** — one self-contained file that\n"
        "  runs in Ollama on any computer.\n"
        "- **B. LoRA adapter (small / 軽い)** — a tiny folder that sits on top of the\n"
        "  `gemma4:e2b` you already have in Ollama."
    ))
    cells.append(md("### A. Merged GGUF / 統合した GGUF\n"
                    "`q4_k_m` is small and fits the free tier. Use `q8_0` for higher\n"
                    "quality if it doesn't run out of memory.\n"
                    "`q4_k_m` は小さく無料枠に収まります。余裕があれば `q8_0` で高品質に。"))
    cells.append(code(
        "model.save_pretrained_gguf(\n"
        '    "gemma_ft_gguf", tokenizer,\n'
        '    quantization_method = "q4_k_m",   # or "q8_0"\n'
        ")\n"
        "\n"
        "# Find the .gguf and give it a simple name / .gguf を探して分かりやすい名前に\n"
        "import glob, shutil, os\n"
        'g = sorted(glob.glob("gemma_ft_gguf/*.gguf"))\n'
        'assert g, "No GGUF produced"\n'
        'shutil.copy(g[0], "my-gemma.gguf")\n'
        'print("GGUF:", os.path.getsize("my-gemma.gguf")//1_000_000, "MB")'
    ))
    cells.append(code(
        "# Write a Modelfile next to the GGUF / GGUF と一緒に Modelfile を作る\n"
        "modelfile = '''FROM ./my-gemma.gguf\n"
        "PARAMETER temperature 1.0\n"
        "PARAMETER top_p 0.95\n"
        "PARAMETER top_k 64\n"
        "'''\n"
        'open("Modelfile", "w").write(modelfile)\n'
        "\n"
        "from google.colab import files\n"
        'files.download("my-gemma.gguf")\n'
        'files.download("Modelfile")'
    ))
    cells.append(md("### B. LoRA adapter only / LoRA アダプターだけ\n"
                    "Tiny. On your computer, point Ollama at it with `FROM gemma4:e2b` +\n"
                    "`ADAPTER`. See `use/Modelfile.example`.\n"
                    "とても軽量。手元では `FROM gemma4:e2b` + `ADAPTER` で使います。"))
    cells.append(code(
        'model.save_pretrained("gemma_lora")\n'
        'tokenizer.save_pretrained("gemma_lora")\n'
        '!cd gemma_lora && zip -qr ../gemma_lora.zip . && cd ..\n'
        "from google.colab import files\n"
        'files.download("gemma_lora.zip")'
    ))

    # ---------------------------------------------------------------- next
    cells.append(md(
        "## 8. Run it at home / 自宅で動かす\n"
        "On your own computer (Ollama installed):\n"
        "自分のパソコンで(Ollama を入れた状態で):\n"
        "\n"
        "```bash\n"
        "# A) merged GGUF — put my-gemma.gguf and Modelfile in one folder, then:\n"
        "ollama create my-gemma -f Modelfile\n"
        'ollama run my-gemma "What time do you open?"\n'
        "```\n"
        "\n"
        "Then **chat in a browser** (open `use/web-chat/index.html`) or **use it in\n"
        "vibe-local** (`use/vibe-local.md`). Full guide: the repo `README.md`.\n"
        "そのあと **ブラウザで会話**(`use/web-chat/index.html`)したり、**vibe-local で\n"
        "使ったり**(`use/vibe-local.md`)できます。詳しくは `README.md`。\n"
        "\n"
        "> If answers seem off, re-export at `q8_0` (step 7A) and make sure your\n"
        "> Modelfile matches. Gemma 4 E2B is brand new, so quality can vary by tool.\n"
        "> 返答が変なときは `q8_0` で書き出し直し、Modelfile を確認してください。"
    ))
    return cells


def main():
    nb = {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = os.path.abspath(NB_PATH)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
