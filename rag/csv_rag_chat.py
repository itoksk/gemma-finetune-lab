#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV-backed RAG chat for small local knowledge bases.

Usage:
  python3 rag/csv_rag_chat.py data.csv --model gemma4:e2b
  python3 rag/csv_rag_chat.py data.csv --query "神戸でラーメンなら？" --no-llm
"""

import argparse
import csv
import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict


USER_KEYS = ["user", "instruction", "question", "prompt", "input"]
ASSISTANT_KEYS = ["assistant", "response", "answer", "completion", "output"]


def clean(text):
    return ("" if text is None else str(text)).strip()


def normalize(text):
    return clean(text).lower()


def find_column(fieldnames, choices):
    lower = {name.lower(): name for name in fieldnames}
    for choice in choices:
        if choice in lower:
            return lower[choice]
    return None


def read_pairs(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        user_col = find_column(reader.fieldnames, USER_KEYS)
        assistant_col = find_column(reader.fieldnames, ASSISTANT_KEYS)
        if not user_col or not assistant_col:
            raise ValueError("CSV needs user/assistant columns.")
        rows = []
        for row in reader:
            user = clean(row.get(user_col))
            assistant = clean(row.get(assistant_col))
            if user and assistant:
                rows.append({"user": user, "assistant": assistant})
        return rows


def extract_fact(answer):
    name = re.search(r"「([^」]+)」", answer)
    feature = re.search(r"特徴：([^）\n]+)", answer)
    url = re.search(r"https?://\S+", answer)
    return {
        "name": name.group(1).strip() if name else "",
        "feature": feature.group(1).strip() if feature else "",
        "url": url.group(0).strip() if url else "",
    }


def extract_category(question):
    patterns = [
        r"おすすめの(.+?)のお店",
        r"神戸で(.+?)を食べるなら",
        r"(.+?)が食べたい",
        r"人気の(.+?)のお店",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
    return ""


def build_documents(rows):
    grouped = {}
    fallback_count = 0

    for row in rows:
        fact = extract_fact(row["assistant"])
        category = extract_category(row["user"])
        key = fact["name"] or f"row-{fallback_count}"
        if not fact["name"]:
            fallback_count += 1

        doc = grouped.setdefault(key, {
            "title": fact["name"] or row["user"],
            "name": fact["name"],
            "feature": fact["feature"],
            "url": fact["url"],
            "categories": set(),
            "questions": [],
            "answers": [],
        })
        if fact["feature"] and not doc["feature"]:
            doc["feature"] = fact["feature"]
        if fact["url"] and not doc["url"]:
            doc["url"] = fact["url"]
        if category:
            doc["categories"].add(category)
        doc["questions"].append(row["user"])
        doc["answers"].append(row["assistant"])

    docs = []
    for doc in grouped.values():
        categories = sorted(doc["categories"])
        answer = doc["answers"][0]
        parts = [
            doc["title"],
            " ".join(categories),
            doc["feature"],
            doc["url"],
            " ".join(doc["questions"]),
            answer,
        ]
        docs.append({
            "title": doc["title"],
            "categories": categories,
            "feature": doc["feature"],
            "url": doc["url"],
            "answer": answer,
            "text": "\n".join(p for p in parts if p),
        })
    return docs


def char_ngrams(text, min_n=2, max_n=3):
    compact = re.sub(r"\s+", "", normalize(text))
    grams = []
    for n in range(min_n, max_n + 1):
        grams.extend(compact[i:i + n] for i in range(max(0, len(compact) - n + 1)))
    return grams


def word_tokens(text):
    return re.findall(r"[a-z0-9]+|[ぁ-んァ-ン一-龥ー]+", normalize(text))


def tokenize(text):
    return word_tokens(text) + char_ngrams(text)


def build_index(docs):
    doc_terms = []
    df = Counter()
    for doc in docs:
        counts = Counter(tokenize(doc["text"]))
        doc_terms.append(counts)
        df.update(counts.keys())

    total = len(docs)
    idf = {
        term: math.log((total + 1) / (freq + 0.5)) + 1.0
        for term, freq in df.items()
    }
    return doc_terms, idf


def retrieve(query, docs, doc_terms, idf, top_k):
    query_terms = Counter(tokenize(query))
    scores = []
    for i, counts in enumerate(doc_terms):
        score = 0.0
        for term, q_count in query_terms.items():
            if term in counts:
                score += q_count * (1.0 + math.log(counts[term])) * idf.get(term, 1.0)
        if score > 0:
            scores.append((score, docs[i]))
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[:top_k]


def make_context(results):
    blocks = []
    for i, (score, doc) in enumerate(results, 1):
        lines = [f"[{i}] {doc['title']}"]
        if doc["categories"]:
            lines.append("ジャンル: " + ", ".join(doc["categories"]))
        if doc["feature"]:
            lines.append("特徴: " + doc["feature"])
        if doc["url"]:
            lines.append("地図: " + doc["url"])
        lines.append("元データ: " + doc["answer"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def ollama_chat(host, model, messages, temperature):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to Ollama at {host}: {e}") from e
    return body.get("message", {}).get("content", "")


def answer_query(args, docs, doc_terms, idf, query):
    results = retrieve(query, docs, doc_terms, idf, args.top_k)
    if not results:
        return "関連する情報がCSV内に見つかりませんでした。"

    context = make_context(results)
    if args.no_llm:
        return context

    messages = [
        {
            "role": "system",
            "content": (
                "あなたは神戸の飲食店案内役です。必ず与えられたCSV検索結果だけを根拠に答えます。"
                "検索結果にない店名、特徴、URLは作らないでください。候補が複数ある場合は複数提示してください。"
            ),
        },
        {
            "role": "user",
            "content": f"質問: {query}\n\nCSV検索結果:\n{context}\n\nこの情報だけで日本語で答えてください。",
        },
    ]
    return ollama_chat(args.host, args.model, messages, args.temperature)


def build_parser():
    parser = argparse.ArgumentParser(description="Chat over a user/assistant CSV with local Ollama RAG.")
    parser.add_argument("csv_path", help="CSV with user/assistant columns")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--top-k", type=int, default=6, help="number of retrieved records")
    parser.add_argument("--temperature", type=float, default=0.1, help="Ollama generation temperature")
    parser.add_argument("--query", help="ask one question and exit")
    parser.add_argument("--no-llm", action="store_true", help="print retrieved context without calling Ollama")
    return parser


def main(argv):
    args = build_parser().parse_args(argv)
    rows = read_pairs(args.csv_path)
    docs = build_documents(rows)
    doc_terms, idf = build_index(docs)
    print(f"Loaded {len(rows)} rows -> {len(docs)} knowledge records.", file=sys.stderr)

    if args.query:
        print(answer_query(args, docs, doc_terms, idf, args.query))
        return 0

    print("質問を入力してください。終了は exit。", file=sys.stderr)
    while True:
        try:
            query = input("\nあなた: ").strip()
        except EOFError:
            print()
            return 0
        if query.lower() in {"exit", "quit"}:
            return 0
        if not query:
            continue
        print("\nAI:", answer_query(args, docs, doc_terms, idf, query))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
