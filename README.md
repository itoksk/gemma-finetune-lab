# gemma-finetune-lab

**Teach a small AI to talk your way — then run it on your own computer.**
Prepare your data, fine-tune **Gemma 4 E2B**, and chat with the result locally
(in your browser, in the terminal, or inside your own app).

**小さな AI に「あなたらしい話し方」を教えて、自分のパソコンで動かそう。**
データを整え、**Gemma 4 E2B** をファインチューニングし、できたモデルとローカルで
会話します(ブラウザ・ターミナル・自作アプリの中で)。

> A companion lab for the Canadian Academy AI course. Non-commercial, for
> education. / Canadian Academy の AI コース用の教材です。非営利・教育目的。

`日本語` ・ `やさしい にほんご` ・ `English` — scroll to your language.

---

## Can I fine-tune locally? / ローカルでできる？ (the honest answer)

Two different things, two different answers:

| | Where it runs / 実行場所 |
| --- | --- |
| **Training** (the heavy GPU part) / 学習(重い処理) | **Best on Google Colab** (free GPU, any laptop). True on-device training depends on your hardware (below). |
| **Using** the finished model (chat, app) / 使う(会話・アプリ) | **Always local** on your own computer, via Ollama. |

### What can train where / どの端末で学習できる？

| Your computer / あなたの端末 | Train / 学習 | Use / 使う |
| --- | --- | --- |
| Apple Silicon Mac, 16 GB+ | **Mac, local (MLX)** or Colab | Ollama (local) |
| Mac (Intel, or under 16 GB) | Colab | Ollama (local) |
| Windows + NVIDIA GPU (8 GB+) | **Windows, local (Unsloth/WSL2)** or Colab | Ollama (local) |
| Windows (no NVIDIA), Chromebook | Colab | Ollama (local) |

> The notebook you may have seen (Unsloth) needs an **NVIDIA GPU**, so it does
> **not** train on an Apple-Silicon Mac. Macs train locally with **MLX** instead.
> But whatever you train on, **you run the result locally** with Ollama.
>
> Unsloth のノートブックは **NVIDIA GPU** が必要なので、Apple Silicon Mac では
> 学習できません。Mac は **MLX** でローカル学習します。どこで学習しても、
> **できたモデルは手元でローカルに動かせます**(Ollama)。

---

## The four steps / 4つのステップ

```
  1. PREPARE        2. TRAIN                3. IMPORT          4. USE
  your data   ->    Colab / Mac / Windows   ->  into Ollama  ->  browser app
  (CSV)             (LoRA)                      (Modelfile)      vibe-local
```

1. **Prepare** examples of how your AI should answer — `scripts/prepare_data.py`.
2. **Train** a LoRA fine-tune — Colab notebook, or locally (Mac/Windows).
3. **Import** the result into Ollama — one `ollama create`.
4. **Use** it — a browser chat app, or vibe-local in the terminal.

---

## 日本語

### これは何？
あなたが書いた会話の例から、Gemma 4 E2B という小さな AI に「答え方」を教えます。
できあがった AI は、あなたのパソコンの中だけで動きます(ネットに送りません)。

### 必要なもの
- **データを作る**: パソコンと Python だけ(インストール不要)。
- **学習する**: 無料の Google Colab(おすすめ)、または Mac/Windows のローカル。
- **使う**: [Ollama](https://ollama.com/download)(無料)。

### 手順

**1) データを用意する**
2列の表(`user` と `assistant`)に例を書きます。`data/template.csv` を真似するか、
Google スプレッドシートで作って CSV 書き出し。詳しくは [`data/README.md`](data/README.md)。

```bash
python3 scripts/prepare_data.py data/examples/cafe_faq.csv
# 性格をつけるなら / add a personality:
python3 scripts/prepare_data.py my_data.csv --system "あなたは親切な案内役です。"
```

**2) 学習する**(下から1つ選ぶ)
- **Colab(だれでも)**: [ノートブックを開く](notebooks/gemma4_e2b_finetune_colab.ipynb) → ランタイムを T4 GPU に → 全部実行。
- **Mac(ローカル)**: [`local/mac_mlx.md`](local/mac_mlx.md) の手順、または `bash local/finetune_mlx.sh data/examples/kobe_guide.csv`。
- **Windows(NVIDIA GPU)**: [`local/windows_unsloth.md`](local/windows_unsloth.md)。

**3) Ollama に取り込む**
学習で `my-gemma.gguf` と `Modelfile` ができます。同じフォルダに置いて:

```bash
bash local/import_to_ollama.sh my-gemma .   # macOS/Linux
# Windows: powershell -File local\import_to_ollama.ps1 my-gemma .
ollama run my-gemma "こんにちは"
```

**4) 使う**
- **ブラウザのアプリ**: [`use/web-chat/`](use/web-chat/) を開く。
- **vibe-local(ターミナル)**: [`use/vibe-local.md`](use/vibe-local.md)。

### 守ってほしいこと
- データに**個人情報を入れない**(実名・住所・電話番号・パスワードなど)。
- 作ったモデルに**人を傷つける・だます**ことをさせない。
- Gemma の[利用規約](https://ai.google.dev/gemma/terms)に従う。

---

## やさしい にほんご

小さな AI に、あなたの「こたえかた」を おしえる ラボです。できた AI は、自分の
パソコンの 中だけで うごきます。

**やること は 4つ**
1. **データを つくる** … 「しつもん」と「こたえ」を 表に かく(`data/template.csv` を まねる)。
2. **がくしゅう する** … 無料の **Google Colab** が かんたん([ノートブック](notebooks/gemma4_e2b_finetune_colab.ipynb))。Mac の人は [`local/mac_mlx.md`](local/mac_mlx.md)。
3. **Ollama に いれる** … `ollama create my-gemma -f Modelfile`。
4. **つかう** … [ブラウザのアプリ](use/web-chat/) や [vibe-local](use/vibe-local.md) で 会話する。

**大事なこと**: データに 名前・住所・電話番号 など、こじんの じょうほうは 入れない。

```bash
# データを つくる
python3 scripts/prepare_data.py data/examples/cafe_faq.csv
```

---

## English

### What is this?
From short examples *you* write, you teach a small AI (Gemma 4 E2B) how to answer.
The finished AI runs entirely on your own computer — nothing is sent online.

### What you need
- **Prepare data**: just a computer and Python (no install).
- **Train**: free Google Colab (recommended), or locally on a Mac/Windows.
- **Use**: [Ollama](https://ollama.com/download) (free).

### Steps

**1) Prepare your data**
Write examples in two columns (`user`, `assistant`). Copy `data/template.csv`, or
build a Google Sheet and export CSV. Full guide: [`data/README.md`](data/README.md).

```bash
python3 scripts/prepare_data.py data/examples/cafe_faq.csv
python3 scripts/prepare_data.py my_data.csv --system "You are a helpful guide."
```

**2) Train** (pick one)
- **Colab (anyone)**: [open the notebook](notebooks/gemma4_e2b_finetune_colab.ipynb) → set runtime to T4 GPU → Run all.
- **Mac (local)**: [`local/mac_mlx.md`](local/mac_mlx.md), or `bash local/finetune_mlx.sh data/examples/kobe_guide.csv`.
- **Windows (NVIDIA GPU)**: [`local/windows_unsloth.md`](local/windows_unsloth.md).

**3) Import into Ollama**
Training gives you `my-gemma.gguf` + a `Modelfile`. Put them in one folder, then:

```bash
bash local/import_to_ollama.sh my-gemma .
ollama run my-gemma "Hello!"
```

**4) Use it**
- **Browser app**: open [`use/web-chat/`](use/web-chat/).
- **vibe-local (terminal)**: see [`use/vibe-local.md`](use/vibe-local.md).

### Responsible use
- **Never put private information** in your data (real names, addresses, phone
  numbers, passwords).
- Don't make a model that harms or deceives people.
- Follow the Gemma [Terms of Use](https://ai.google.dev/gemma/terms).

---

## Repository layout / 構成

```
data/        template + example datasets + how to design good data
scripts/     prepare_data.py (data tool) + build_notebook.py (regenerate the notebook)
notebooks/   gemma4_e2b_finetune_colab.ipynb  (Colab training)
local/       mac_mlx.md, windows_unsloth.md, finetune_mlx.sh, import_to_ollama.sh/.ps1
use/         Modelfile.example, web-chat/ (browser app), vibe-local.md
```

## Troubleshooting / 困ったとき

| Symptom / 症状 | Fix / 対処 |
| --- | --- |
| Colab: no GPU / GPU が無い | Runtime ▸ Change runtime type ▸ T4 GPU |
| Replies seem off / 返答が変 | Re-export at `q8_0` (notebook step 7), check the `Modelfile` template. Gemma 4 E2B is new — quality varies by tool. |
| `ollama: command not found` | Install Ollama: https://ollama.com/download |
| Browser app won't connect | Serve the folder (`python3 -m http.server`) and set `OLLAMA_ORIGINS=*`, restart Ollama. See [`use/web-chat/README.md`](use/web-chat/README.md). |
| Mac: `mlx_lm.lora` can't load Gemma 4 | See [`local/mac_mlx.md`](local/mac_mlx.md) Troubleshooting (try `mlx-vlm`, or use Colab). |

---

Built on [Unsloth](https://github.com/unslothai/unsloth) (Colab/Windows),
[MLX](https://github.com/ml-explore/mlx-lm) (Mac), and
[Ollama](https://ollama.com). Model: Google **Gemma 4 E2B**.
