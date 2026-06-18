# Designing your training data / 学習データの作り方

The fine-tuned model is only as good as the examples you give it. This is the
most important step — spend real time here.

ファインチューニングしたモデルの質は、あなたが与える例で決まります。ここが
いちばん大切なステップです。じっくり取り組みましょう。

---

## 1. What "training data" is / 「学習データ」とは

A list of small conversations: **what the user says**, and **how you want your
AI to answer**. The model learns the *pattern and voice* of your answers.

小さな会話のリストです。**ユーザーが言うこと**と、**AI にどう答えてほしいか**
の組。モデルはあなたの答え方の*パターンと口調*を学びます。

The easiest format is a CSV with two columns — open `template.csv` and start
typing, or make a Google Sheet with the same two columns and export as CSV.

いちばん簡単なのは、2列の CSV です。`template.csv` を開いて書き始めるか、
同じ2列の Google スプレッドシートを作って CSV で書き出してください。

| user | assistant |
| --- | --- |
| What time do you open? | We open at 8:00 every weekday morning. |
| Do you have anything vegan? | Yes! Our oat-milk latte and fruit cups are vegan. |

> The script also accepts column names `instruction,response`,
> `question,answer`, `prompt,completion`, or `input,output`.
> An optional `system` column gives a row its own personality.
>
> 列名は `instruction,response` / `question,answer` / `prompt,completion` /
> `input,output` でも構いません。`system` 列で1行ごとの性格も指定できます。

---

## 2. How many examples? / 何件くらい必要？

| count / 件数 | result / 結果 |
| --- | --- |
| under 10 / 10未満 | the model barely changes / ほとんど変わりません |
| 30–50 | a noticeable personality / 性格が少し見えてきます |
| **100–300** | **a clear, reliable style — a good target** / **はっきり安定。おすすめ** |
| 1000+ | great, but a lot of writing / とても良いが、書くのが大変 |

Quality beats quantity. 50 careful examples beat 300 sloppy ones.

量より質です。ていねいな50件は、雑な300件に勝ります。

---

## 3. Five rules for good examples / 良い例の5つのルール

1. **Be consistent / 一貫性をもたせる** — Always answer in the same voice
   (friendly? formal? short?). Mixed voices confuse the model.
   いつも同じ口調で答える。口調がバラバラだとモデルは混乱します。
2. **Be diverse / 多様にする** — Ask the same thing in different words
   ("What time do you open?" / "when r u open" / "opening hours?").
   同じことを別の言い方でも聞く。表現にばらつきを持たせます。
3. **Sound real / 自然な言い回し** — Write the way students actually type,
   including casual or short messages.
   実際に打つような自然な文で。くだけた短文も入れましょう。
4. **Answer clearly / 答えははっきり** — Give the exact answer you want every
   time, not "maybe" or "I think".
   毎回、望む答えをはっきり書く。「たぶん」は避けます。
5. **No private info / 個人情報は入れない** — Never put real names, phone
   numbers, addresses, passwords, or anything personal in the data.
   実名・電話番号・住所・パスワードなど個人情報は絶対に入れない。

---

## 4. Do / Don't / して良いこと・ダメなこと

| Do / ○ | Don't / × |
| --- | --- |
| Keep answers short and useful | Write essays the model can't imitate |
| Cover the questions people really ask | Only cover one topic 100 times |
| Re-use the same persona everywhere | Switch tone between rows |
| Double-check facts in your answers | Teach the model wrong facts |
| Remove duplicates (the script does too) | Paste the same row 50 times |

---

## 5. Prepare and check your file / ファイルを変換して点検する

```bash
# Build training data (writes to out/) / 学習データを作る（out/ に出力）
python3 scripts/prepare_data.py data/examples/cafe_faq.csv

# Add a personality / 性格を与える
python3 scripts/prepare_data.py my_data.csv --system "You are a calm study buddy."

# Merge exact duplicate questions with multiple valid answers.
# 同じ質問に複数の正しい答えがある場合は、候補一覧の1例にまとめる
python3 scripts/prepare_data.py my_restaurants.csv --merge-same-user

# Check a file you already made / すでに作ったファイルを点検する
python3 scripts/prepare_data.py validate out/train_sharegpt.jsonl
```

The script tells you how many examples it kept, removes duplicates, warns about
problems, and writes the files the training notebook needs.

スクリプトは、採用した件数を表示し、重複を取り除き、問題があれば警告し、
学習ノートブックが必要とするファイルを書き出します。

It creates two outputs from one source — you only edit your CSV:

ひとつの元データから2種類を作ります。あなたは CSV を編集するだけです:

- `out/train_sharegpt.jsonl` → the **Google Colab** notebook (Unsloth)
- `out/train.jsonl` + `out/valid.jsonl` → **local Mac** training (MLX)

### Same question, many correct answers / 同じ質問に複数の正解があるとき

Do not train rows like this as separate single-answer examples:

```csv
user,assistant
Where should I eat ramen in Kobe?,Metro Ramen is good.
Where should I eat ramen in Kobe?,Mokkosu is good.
Where should I eat ramen in Kobe?,Ramen Taro is good.
```

That teaches contradictory answers. Use `--merge-same-user` so the tool creates
one multi-candidate answer, or rewrite the questions to be more specific.

これは矛盾した答えを教えることになります。`--merge-same-user` で候補一覧の
1回答にまとめるか、質問を具体的に分けてください。

```bash
python3 scripts/prepare_data.py my_restaurants.csv --merge-same-user
```

If you need exact factual lookup from a CSV (shop names, features, map URLs),
use the RAG tool instead of relying on memorization:

```bash
python3 rag/csv_rag_chat.py my_restaurants.csv --query "神戸でラーメンなら？" --no-llm
```

## 6. Getting your data into the notebook / ノートブックへの渡し方

The Colab notebook can take your data **two ways** (set `DATA_SOURCE`):

Colab ノートブックは、あなたのデータを **2通り** で受け取れます(`DATA_SOURCE` で選ぶ):

1. **Upload / アップロード** — choose your CSV or `.jsonl` from your computer.
   パソコンから CSV か `.jsonl` を選ぶ。
2. **From your GitHub / GitHub から** — put your file in your own repo, open it,
   click **Raw**, copy the URL, and paste it into `MY_DATA_URL`. The notebook
   downloads it automatically.
   自分のリポジトリにファイルを置き、開いて **Raw** を押し、その URL を
   `MY_DATA_URL` に貼る。ノートブックが自動で取得します。

Either way, a **CSV** is auto-converted by `prepare_data.py` inside the notebook,
and a `.jsonl` is used as-is. **CSV** はノート内で自動変換、`.jsonl` はそのまま。

---

## Example datasets here / ここにある例

- `examples/cafe_faq.csv` — a friendly school-café FAQ bot (32 examples).
- `examples/kobe_guide.csv` — a Kobe tour-guide persona (30 examples). Try it
  with `--system "You are a friendly Kobe tour guide."`

Copy one, change the answers to fit *your* idea, and make it your own.

どれかをコピーして、答えを*あなたのアイデア*に書き換え、自分だけのデータに
しましょう。
