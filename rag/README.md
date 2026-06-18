# CSV RAG chat / CSV検索チャット

Fine-tuning teaches a model a style and rough behavior. RAG is better when the
answer must stay tied to a concrete CSV, such as restaurant names, features, and
map URLs.

ファインチューニングは「答え方」を教えるのに向いています。店名・特徴・地図URLの
ように、CSVの事実に基づいて答えたい場合は RAG の方が安定します。

## Try retrieval only / 検索だけ試す

```bash
python3 rag/csv_rag_chat.py /Users/keisuke/Downloads/kobe_restaurants_clean.csv \
  --query "神戸でラーメンなら？" \
  --no-llm
```

This does not call Ollama. It only prints the CSV rows that would be used as
context.

## Chat with Ollama / Ollamaで回答する

Start Ollama and make sure the model exists:

```bash
ollama pull gemma4:e2b
python3 rag/csv_rag_chat.py /Users/keisuke/Downloads/kobe_restaurants_clean.csv \
  --model gemma4:e2b \
  --query "神戸でラーメンなら？"
```

Interactive mode:

```bash
python3 rag/csv_rag_chat.py /Users/keisuke/Downloads/kobe_restaurants_clean.csv \
  --model gemma4:e2b
```

Type `exit` to quit.

## When to use this instead of fine-tuning / 使い分け

- Use **RAG** for facts: shop names, menus, features, URLs, rules, prices.
- Use **fine-tuning** for tone: friendly guide, short answers, classroom style.
- Best result: keep the base or fine-tuned model, but answer factual questions
  through this RAG script.

