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
        "Your data can come from **three** places. Set `DATA_SOURCE` in the next\n"
        "cell, then run.\n"
        "データの取り込み方は **3通り**。次のセルの `DATA_SOURCE` を選んで実行します。\n"
        "\n"
        "- `\"upload\"` — pick a file from your computer (a CSV or a .jsonl)\n"
        "  パソコンからファイルを選ぶ(CSV か .jsonl)\n"
        "- `\"github\"` — load your own file from **your** GitHub (paste its Raw URL)\n"
        "  **自分の** GitHub に置いたファイルを読む(Raw の URL を貼る)\n"
        "- `\"example\"` — try a ready-made example dataset / 用意された例で試す\n"
        "\n"
        "A **CSV** is auto-cleaned + converted by `prepare_data.py`; a **.jsonl**\n"
        "(already prepared) is used as-is.\n"
        "**CSV** は `prepare_data.py` が自動で整形・変換、**.jsonl** はそのまま使います。\n"
        "\n"
        "If your CSV has the **same question with different valid answers** (for\n"
        "example, one food genre with many restaurant candidates), keep\n"
        "`MERGE_SAME_USER = True`. It turns those rows into one multi-candidate\n"
        "answer instead of teaching contradictory single answers.\n"
        "同じ質問に複数の正しい答えがある CSV（例: 1ジャンルに複数店舗）では\n"
        "`MERGE_SAME_USER = True` のままにします。矛盾した単一回答ではなく、\n"
        "候補一覧の1回答にまとめます。"
    ))
    cells.append(code(
        '# Choose ONE / どれか1つ:  "upload" | "github" | "example"\n'
        'DATA_SOURCE = "upload"\n'
        "\n"
        "# Merge exact duplicate questions into one multi-answer training example.\n"
        "# 同じ質問に複数回答がある場合、候補一覧の1例にまとめる。\n"
        "MERGE_SAME_USER = True\n"
        "\n"
        '# For "github" only: paste your file\'s RAW url\n'
        '# (open the file on GitHub, click the "Raw" button, copy that URL).\n'
        '# 「github」のときだけ: GitHub でファイルを開き "Raw" を押した URL を貼る。\n'
        'MY_DATA_URL = ""  # e.g. https://raw.githubusercontent.com/you/your-repo/main/mydata.csv'
    ))
    cells.append(code(
        "import csv, os, re, subprocess, sys, urllib.request\n"
        "from collections import OrderedDict\n"
        f'LAB_RAW = "{RAW}"\n'
        "\n"
        "def _clean_text(value):\n"
        "    return (\"\" if value is None else str(value)).strip()\n"
        "\n"
        "def _extract_candidate(answer):\n"
        "    name = re.search(r\"「([^」]+)」\", answer)\n"
        "    feature = re.search(r\"特徴：([^）\\n]+)\", answer)\n"
        "    url = re.search(r\"https?://\\S+\", answer)\n"
        "    if not name:\n"
        "        return None\n"
        "    return {\n"
        "        \"name\": name.group(1).strip(),\n"
        "        \"feature\": feature.group(1).strip() if feature else \"\",\n"
        "        \"url\": url.group(0).strip() if url else \"\",\n"
        "    }\n"
        "\n"
        "def _format_merged_answer(answers):\n"
        "    unique = list(OrderedDict((a, None) for a in answers if _clean_text(a)).keys())\n"
        "    candidates, seen = [], set()\n"
        "    for answer in unique:\n"
        "        item = _extract_candidate(answer)\n"
        "        if item and item[\"name\"] not in seen:\n"
        "            candidates.append(item)\n"
        "            seen.add(item[\"name\"])\n"
        "    if len(candidates) == len(unique):\n"
        "        lines = [\"候補は複数あります。おすすめは次のお店です。\"]\n"
        "        for i, item in enumerate(candidates, 1):\n"
        "            line = f\"{i}. {item['name']}\"\n"
        "            if item[\"feature\"]:\n"
        "                line += f\"（特徴：{item['feature']}）\"\n"
        "            if item[\"url\"]:\n"
        "                line += f\"\\n   地図: {item['url']}\"\n"
        "            lines.append(line)\n"
        "        return \"\\n\".join(lines)\n"
        "    lines = [\"候補は複数あります。代表的な回答は次のとおりです。\"]\n"
        "    for i, answer in enumerate(unique, 1):\n"
        "        lines.append(f\"{i}. {answer}\")\n"
        "    return \"\\n\".join(lines)\n"
        "\n"
        "def merge_same_user_csv(path):\n"
        "    with open(path, newline=\"\", encoding=\"utf-8-sig\") as f:\n"
        "        reader = csv.DictReader(f)\n"
        "        rows = list(reader)\n"
        "        fieldnames = reader.fieldnames or []\n"
        "    lower = {name.lower(): name for name in fieldnames}\n"
        "    user_col = next((lower[k] for k in [\"user\", \"instruction\", \"question\", \"prompt\", \"input\"] if k in lower), None)\n"
        "    assistant_col = next((lower[k] for k in [\"assistant\", \"response\", \"answer\", \"completion\", \"output\"] if k in lower), None)\n"
        "    system_col = next((lower[k] for k in [\"system\", \"persona\", \"instructions\"] if k in lower), None)\n"
        "    if not user_col or not assistant_col:\n"
        "        return path\n"
        "    groups = OrderedDict()\n"
        "    for row in rows:\n"
        "        user = _clean_text(row.get(user_col))\n"
        "        assistant = _clean_text(row.get(assistant_col))\n"
        "        system = _clean_text(row.get(system_col)) if system_col else \"\"\n"
        "        if not user or not assistant:\n"
        "            continue\n"
        "        groups.setdefault((system, user), []).append(assistant)\n"
        "    out_rows, merged_groups, merged_input_rows = [], 0, 0\n"
        "    for (system, user), answers in groups.items():\n"
        "        unique = list(OrderedDict((a, None) for a in answers).keys())\n"
        "        if len(unique) > 1:\n"
        "            assistant = _format_merged_answer(unique)\n"
        "            merged_groups += 1\n"
        "            merged_input_rows += len(answers)\n"
        "        else:\n"
        "            assistant = unique[0]\n"
        "        rec = {\"user\": user, \"assistant\": assistant}\n"
        "        if system:\n"
        "            rec[\"system\"] = system\n"
        "        out_rows.append(rec)\n"
        "    if not merged_groups:\n"
        "        print(\"No duplicate questions to merge / 集約する重複質問はありません\")\n"
        "        return path\n"
        "    out = os.path.splitext(path)[0] + \"_merged.csv\"\n"
        "    out_fields = [\"user\", \"assistant\"] + ([\"system\"] if any(\"system\" in r for r in out_rows) else [])\n"
        "    with open(out, \"w\", newline=\"\", encoding=\"utf-8\") as f:\n"
        "        writer = csv.DictWriter(f, fieldnames=out_fields)\n"
        "        writer.writeheader()\n"
        "        writer.writerows(out_rows)\n"
        "    print(f\"Merged duplicate questions: {merged_groups} groups, {merged_input_rows} rows -> {len(out_rows)} rows\")\n"
        "    print(\"集約後CSV:\", out)\n"
        "    return out\n"
        "\n"
        'if DATA_SOURCE == "upload":\n'
        "    from google.colab import files\n"
        '    print("Choose your file (CSV or .jsonl) / ファイルを選ぶ")\n'
        "    up = files.upload(); src = next(iter(up))\n"
        'elif DATA_SOURCE == "github":\n'
        '    assert MY_DATA_URL, "Set MY_DATA_URL first / さきに MY_DATA_URL を設定"\n'
        '    src = MY_DATA_URL.split("/")[-1].split("?")[0]\n'
        "    urllib.request.urlretrieve(MY_DATA_URL, src)\n"
        "else:  # example\n"
        '    urllib.request.urlretrieve(LAB_RAW + "/data/examples/cafe_faq.csv", "cafe_faq.csv")\n'
        '    src = "cafe_faq.csv"\n'
        'print("Got:", src)\n'
        "\n"
        "# A CSV is cleaned + converted by our data tool; a .jsonl is ready already.\n"
        "# CSV は整形ツールで変換、.jsonl はそのまま。\n"
        'if src.endswith(".csv"):\n'
        '    if MERGE_SAME_USER:\n'
        '        src = merge_same_user_csv(src)\n'
        '    urllib.request.urlretrieve(LAB_RAW + "/scripts/prepare_data.py", "prepare_data.py")\n'
        '    cmd = [sys.executable, "prepare_data.py", src]\n'
        '    r = subprocess.run(cmd, capture_output=True, text=True)\n'
        '    print(r.stdout)\n'
        '    if r.returncode != 0:\n'
        '        print(r.stderr)\n'
        '        raise RuntimeError(f"prepare_data.py failed / 失敗しました: exit={r.returncode}")\n'
        '    data_file = "out/train_sharegpt.jsonl"\n'
        "else:\n"
        "    data_file = src\n"
        'print("data_file =", data_file)'
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
        "Training runs for a number of **epochs** (full passes over your data). The\n"
        "fewer examples you have, the more epochs you need for the personality to\n"
        "stick:\n"
        "**エポック**(データを丸ごと何周するか)で学習します。例が少ないほど、\n"
        "個性を定着させるには多くのエポックが必要です:\n"
        "\n"
        "- ~30-60 examples / 例が30〜60件   -> 8-10\n"
        "- ~100-300 examples / 100〜300件   -> 4-6\n"
        "- 500+ examples / 500件以上        -> 2-3\n"
        "\n"
        "Too few and nothing changes; way too many just memorises. Watch the loss\n"
        "drop. 少なすぎると変化なし、多すぎると丸暗記。loss が下がるのを見ます。"
    ))
    cells.append(code(
        "EPOCHS = 8   # 30-60 ex: 8-10 | 100-300: 4-6 | 500+: 2-3\n"
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
    cells.append(code(
        "# Quick mask check: learned tokens must be > 0.\n"
        "# 確認: learned tokens が 0 なら返答部分を学習できていません。\n"
        "batch = next(iter(trainer.get_train_dataloader()))\n"
        'labels = batch["labels"]\n'
        "learned_tokens = int((labels != -100).sum())\n"
        'print("learned tokens:", learned_tokens)\n'
        'assert learned_tokens > 0, "No assistant tokens are being trained. Check chat template markers."\n'
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
        '    "content": [{"type": "text", "text": "神戸でおすすめのラーメンのお店はありますか？"}]\n'
        "}]\n"
        "inputs = tokenizer.apply_chat_template(\n"
        "    messages,\n"
        "    add_generation_prompt=True,\n"
        "    enable_thinking=False,\n"
        "    return_tensors='pt',\n"
        "    tokenize=True, return_dict=True,\n"
        ').to("cuda")\n'
        "_ = model.generate(**inputs, max_new_tokens=256,\n"
        "                   do_sample=False,\n"
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
        "PARAMETER temperature 0.1\n"
        "PARAMETER top_p 0.9\n"
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
