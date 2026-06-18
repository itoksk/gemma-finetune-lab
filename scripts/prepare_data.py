#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py — turn your raw examples into clean training data for Gemma 4 E2B.
あなたの作った例を、Gemma 4 E2B 学習用のきれいなデータに変換します。

No installation needed. Pure Python standard library.
インストール不要。Python 標準ライブラリだけで動きます。

------------------------------------------------------------------------------
WHAT IT DOES / なにをするか
------------------------------------------------------------------------------
You write examples of "when the user says X, the AI should answer Y".
「ユーザーが X と言ったら、AI は Y と答える」という例をあなたが書きます。
This script reads them, checks them, and writes two kinds of files:
このスクリプトはそれを読み、チェックして、2種類のファイルを書き出します:

  out/train_sharegpt.jsonl   -> for the Google Colab notebook (Unsloth)
                                 Google Colab ノートブック (Unsloth) 用
  out/train.jsonl            -> for local training on a Mac (MLX)
  out/valid.jsonl               Mac でのローカル学習 (MLX) 用

------------------------------------------------------------------------------
HOW TO USE / つかいかた
------------------------------------------------------------------------------
  # 1) Build training data from a CSV / CSV から学習データを作る
  python3 scripts/prepare_data.py data/examples/cafe_faq.csv

  # Give your AI a personality (a "system" instruction) / 性格(system命令)を与える
  python3 scripts/prepare_data.py data/examples/kobe_guide.csv \
      --system "You are a friendly Kobe tour guide."

  # Merge rows that ask the exact same question into one multi-answer example.
  # 同じ質問に複数の答えがある行を、候補一覧の1例にまとめる
  python3 scripts/prepare_data.py my_restaurants.csv --merge-same-user

  # 2) Check a file you already have / すでにあるファイルを点検する
  python3 scripts/prepare_data.py validate out/train_sharegpt.jsonl

------------------------------------------------------------------------------
INPUT FORMATS IT UNDERSTANDS / 読み込める入力フォーマット
------------------------------------------------------------------------------
  * CSV with a question column and an answer column. Any of these pairs work
    (column names are case-insensitive):
      user,assistant   /   instruction,response   /   question,answer
      prompt,completion   /   input,output
    An optional `system` column sets a per-row personality.
  * JSONL / JSON, where each item is one of:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}
      {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", ...}]}
      {"user": "...", "assistant": "..."}   (or instruction/response, etc.)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Small bilingual print helpers / 小さな二言語表示ヘルパー
# ---------------------------------------------------------------------------
RESET = "\033[0m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def _c(code, text):
    # Colour only when writing to a terminal / 端末のときだけ色を付ける
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def ok(msg):
    print(_c(GREEN, "OK   ") + msg)


def warn(msg):
    print(_c(YELLOW, "WARN ") + msg)


def err(msg):
    print(_c(RED, "ERROR") + " " + msg, file=sys.stderr)


def info(msg):
    print(_c(CYAN, "..   ") + msg)


# ---------------------------------------------------------------------------
# Column-name aliases for CSV / CSV の列名エイリアス
# ---------------------------------------------------------------------------
USER_KEYS = ["user", "instruction", "question", "prompt", "input"]
ASSISTANT_KEYS = ["assistant", "response", "answer", "completion", "output"]
SYSTEM_KEYS = ["system", "persona", "instructions"]


def estimate_tokens(text):
    """Rough token estimate that works for English + Japanese.
    英語と日本語の両方で大まかに使えるトークン数の見積もり。
    Latin text ~ 1 token per 4 chars; CJK ~ 1 token per char (a safe upper bound)."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other_chars = len(text) - ascii_chars
    return int(ascii_chars / 4) + other_chars


# ---------------------------------------------------------------------------
# Normalisation: everything becomes a list of "messages"
# 正規化: すべてを messages のリストにそろえる
#   message = {"role": "system"|"user"|"assistant", "content": str}
# ---------------------------------------------------------------------------
def _clean(s):
    return ("" if s is None else str(s)).strip()


def _pair_to_messages(user, assistant, system=None):
    msgs = []
    if _clean(system):
        msgs.append({"role": "system", "content": _clean(system)})
    msgs.append({"role": "user", "content": _clean(user)})
    msgs.append({"role": "assistant", "content": _clean(assistant)})
    return msgs


def _sharegpt_to_messages(convo):
    role_map = {"human": "user", "user": "user", "gpt": "assistant",
                "assistant": "assistant", "system": "system"}
    msgs = []
    for turn in convo:
        frm = role_map.get(_clean(turn.get("from")).lower())
        if frm is None:
            continue
        msgs.append({"role": frm, "content": _clean(turn.get("value"))})
    return msgs


def _record_to_messages(rec, default_system=None):
    """Turn ONE raw record (dict) into a messages list, or None if unusable."""
    if not isinstance(rec, dict):
        return None

    # Already in chat / messages form
    if isinstance(rec.get("messages"), list):
        msgs = [{"role": _clean(m.get("role")).lower(), "content": _clean(m.get("content"))}
                for m in rec["messages"] if isinstance(m, dict)]
    # ShareGPT "conversations"
    elif isinstance(rec.get("conversations"), list):
        msgs = _sharegpt_to_messages(rec["conversations"])
    else:
        # Flat pair: find a user-ish key and an assistant-ish key
        lower = {k.lower(): v for k, v in rec.items()}
        ukey = next((k for k in USER_KEYS if k in lower), None)
        akey = next((k for k in ASSISTANT_KEYS if k in lower), None)
        if ukey is None or akey is None:
            return None
        skey = next((k for k in SYSTEM_KEYS if k in lower), None)
        system = lower.get(skey) if skey else None
        msgs = _pair_to_messages(lower[ukey], lower[akey], system)

    # Inject a default system message if the row has none and one was requested
    if default_system and not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": _clean(default_system)}] + msgs
    return msgs


def _messages_to_record(msgs):
    """Turn a simple chat back into a flat record for post-processing."""
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    body = [m for m in msgs if m["role"] != "system"]
    if len(body) != 2 or body[0]["role"] != "user" or body[1]["role"] != "assistant":
        return None
    rec = {"user": body[0]["content"], "assistant": body[1]["content"]}
    if system:
        rec["system"] = system
    return rec


def _extract_named_candidate(answer):
    """Extract compact restaurant-style facts when the answer follows the lab CSV pattern."""
    name = re.search(r"「([^」]+)」", answer)
    feature = re.search(r"特徴：([^）\n]+)", answer)
    url = re.search(r"https?://\S+", answer)
    if not name:
        return None
    return {
        "name": name.group(1).strip(),
        "feature": feature.group(1).strip() if feature else "",
        "url": url.group(0).strip() if url else "",
    }


def _format_merged_assistant(answers):
    unique = list(OrderedDict((a, None) for a in answers if _clean(a)).keys())
    candidates = []
    seen_names = set()
    for answer in unique:
        item = _extract_named_candidate(answer)
        if item and item["name"] not in seen_names:
            candidates.append(item)
            seen_names.add(item["name"])

    if len(candidates) == len(unique):
        lines = ["候補は複数あります。おすすめは次のお店です。"]
        for i, item in enumerate(candidates, 1):
            line = f"{i}. {item['name']}"
            if item["feature"]:
                line += f"（特徴：{item['feature']}）"
            if item["url"]:
                line += f"\n   地図: {item['url']}"
            lines.append(line)
        return "\n".join(lines)

    lines = ["候補は複数あります。代表的な回答は次のとおりです。"]
    for i, answer in enumerate(unique, 1):
        lines.append(f"{i}. {answer}")
    return "\n".join(lines)


def merge_same_user_records(records, default_system=None):
    """Merge exact duplicate user prompts that have different assistant answers."""
    groups = OrderedDict()
    passthrough = []

    for rec in records:
        msgs = _record_to_messages(rec, default_system)
        flat = _messages_to_record(msgs) if msgs else None
        if flat is None:
            passthrough.append(rec)
            continue
        key = (_clean(flat.get("system")), flat["user"])
        if key not in groups:
            groups[key] = []
        groups[key].append(flat)

    merged = []
    merged_groups = 0
    merged_rows = 0

    for (system, user), rows in groups.items():
        answers = [r["assistant"] for r in rows]
        unique_answers = list(OrderedDict((a, None) for a in answers if _clean(a)).keys())
        if len(unique_answers) <= 1:
            rec = {"user": user, "assistant": unique_answers[0] if unique_answers else ""}
        else:
            rec = {"user": user, "assistant": _format_merged_assistant(unique_answers)}
            merged_groups += 1
            merged_rows += len(rows)
        if system:
            rec["system"] = system
        merged.append(rec)

    return passthrough + merged, {
        "merged_prompt_groups": merged_groups,
        "merged_input_rows": merged_rows,
        "rows_after_merge": len(passthrough) + len(merged),
    }


# ---------------------------------------------------------------------------
# Readers / 読み込み
# ---------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row / CSV にヘッダー行がありません")
        rows = [dict(r) for r in reader]
    return rows


def read_jsonl_or_json(path):
    text = open(path, encoding="utf-8").read().strip()
    if not text:
        return []
    # Try a single JSON document first (could be a list) / まず JSON 全体として試す
    try:
        doc = json.loads(text)
        if isinstance(doc, list):
            return doc
        if isinstance(doc, dict):
            return [doc]
    except json.JSONDecodeError:
        pass
    # Fall back to JSON Lines / JSON Lines として読む
    records = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {i} is not valid JSON / {i}行目が壊れています: {e}")
    return records


def load_records(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_csv(path)
    if ext in (".jsonl", ".json", ".ndjson"):
        return read_jsonl_or_json(path)
    raise ValueError(f"Unknown file type '{ext}'. Use .csv, .jsonl or .json / "
                     f"未知の拡張子 '{ext}'。.csv / .jsonl / .json を使ってください")


# ---------------------------------------------------------------------------
# Validation / 検証
# ---------------------------------------------------------------------------
VALID_ROLES = {"system", "user", "assistant"}


def validate_messages(msgs):
    """Return list of human-readable problems (empty list == fine)."""
    problems = []
    if not msgs:
        return ["empty conversation / 会話が空です"]
    body = [m for m in msgs if m["role"] != "system"]
    for m in msgs:
        if m["role"] not in VALID_ROLES:
            problems.append(f"unknown role '{m['role']}' / 不明なrole")
        if not m["content"]:
            problems.append(f"empty {m['role']} message / {m['role']} の中身が空です")
    if not body:
        problems.append("no user/assistant turns / user・assistant の発話がありません")
        return problems
    if body[0]["role"] != "user":
        problems.append("conversation must start with the user / 会話は user から始めます")
    if body[-1]["role"] != "assistant":
        problems.append("conversation must end with the assistant / 会話は assistant で終えます")
    # Check strict alternation user, assistant, user, assistant ...
    for idx, m in enumerate(body):
        want = "user" if idx % 2 == 0 else "assistant"
        if m["role"] != want:
            problems.append("turns must alternate user/assistant / "
                            "user と assistant を交互にします")
            break
    return problems


def dedup_key(msgs):
    return json.dumps(msgs, ensure_ascii=False, sort_keys=True)


def normalise_and_validate(records, default_system=None):
    """Return (good_conversations, stats, fatal_error_or_None)."""
    good = []
    seen = set()
    skipped, duplicates = 0, 0
    for i, rec in enumerate(records, 1):
        msgs = _record_to_messages(rec, default_system)
        if msgs is None:
            skipped += 1
            warn(f"row {i}: could not find a question/answer pair — skipped / "
                 f"{i}行目: 質問と答えの組が見つからず、スキップしました")
            continue
        problems = validate_messages(msgs)
        if problems:
            skipped += 1
            warn(f"row {i}: {problems[0]} — skipped / スキップしました")
            continue
        key = dedup_key(msgs)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        good.append(msgs)
    stats = compute_stats(good)
    stats.update({"skipped": skipped, "duplicates": duplicates,
                  "input_rows": len(records)})
    return good, stats, None


def compute_stats(conversations):
    char_lengths, tok_lengths = [], []
    for msgs in conversations:
        total = sum(len(m["content"]) for m in msgs)
        char_lengths.append(total)
        tok_lengths.append(sum(estimate_tokens(m["content"]) for m in msgs))
    char_lengths.sort()
    tok_lengths.sort()

    def median(xs):
        return 0 if not xs else xs[len(xs) // 2]

    n = len(conversations)
    return {
        "count": n,
        "avg_chars": int(sum(char_lengths) / n) if n else 0,
        "median_chars": median(char_lengths),
        "max_chars": char_lengths[-1] if char_lengths else 0,
        "avg_tokens": int(sum(tok_lengths) / n) if n else 0,
        "max_tokens": tok_lengths[-1] if tok_lengths else 0,
    }


def print_stats(stats, max_seq_length):
    print()
    print(_c(DIM, "  ------------- dataset stats / データの統計 -------------"))
    print(f"  examples kept / 採用した例 : {stats['count']}")
    if stats.get("input_rows") is not None:
        print(f"  read from input / 読み込み : {stats['input_rows']}")
    if stats.get("duplicates"):
        print(f"  duplicates removed / 重複削除: {stats['duplicates']}")
    if stats.get("skipped"):
        print(f"  skipped (bad rows) / 除外  : {stats['skipped']}")
    print(f"  avg / median / max chars   : "
          f"{stats['avg_chars']} / {stats['median_chars']} / {stats['max_chars']}")
    print(f"  avg / max est. tokens      : {stats['avg_tokens']} / {stats['max_tokens']}")
    print(_c(DIM, "  -------------------------------------------------------"))
    print()

    # Friendly guidance / やさしいアドバイス
    if stats["count"] == 0:
        err("No usable examples. Check your file format. / "
            "使える例が0件です。ファイル形式を確認してください。")
    elif stats["count"] < 10:
        warn(f"Only {stats['count']} examples. The model will barely change. "
             f"Aim for 50+. / 例が{stats['count']}件だけです。50件以上を目標に。")
    elif stats["count"] < 30:
        warn(f"{stats['count']} examples is a small start — 50-200 works much "
             f"better. / {stats['count']}件は少なめ。50〜200件が良いです。")
    else:
        ok(f"{stats['count']} examples — a healthy size to start with. / "
           f"{stats['count']}件。学習を始めるのに十分です。")

    if stats["max_tokens"] > max_seq_length:
        warn(f"Longest example ~{stats['max_tokens']} tokens > max_seq_length "
             f"{max_seq_length}. It will be truncated; shorten it or raise "
             f"max_seq_length. / いちばん長い例が長すぎます。短くするか "
             f"max_seq_length を上げてください。")


# ---------------------------------------------------------------------------
# Writers / 書き出し
# ---------------------------------------------------------------------------
def to_sharegpt(msgs):
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    return {"conversations": [{"from": role_map[m["role"]], "value": m["content"]}
                              for m in msgs]}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_valid(conversations, valid_frac, seed):
    """Deterministic shuffle + split. 決定的なシャッフルと分割。"""
    import random
    rng = random.Random(seed)
    items = list(conversations)
    rng.shuffle(items)
    n = len(items)
    if n < 2:
        # Too small to split — reuse the single example for validation too.
        return items, items[:]
    n_valid = max(1, int(round(n * valid_frac)))
    n_valid = min(n_valid, n - 1)  # always keep at least 1 for training
    return items[n_valid:], items[:n_valid]


# ---------------------------------------------------------------------------
# Commands / コマンド
# ---------------------------------------------------------------------------
def cmd_build(args):
    if not os.path.isfile(args.input):
        err(f"File not found / ファイルが見つかりません: {args.input}")
        return 2
    info(f"Reading {args.input} ...")
    try:
        records = load_records(args.input)
    except ValueError as e:
        err(str(e))
        return 2

    merge_stats = None
    if args.merge_same_user:
        records, merge_stats = merge_same_user_records(records, default_system=args.system)
        if merge_stats["merged_prompt_groups"]:
            ok("merged exact duplicate user prompts into multi-answer examples / "
               "同じ質問を候補一覧の例にまとめました")
            print(f"  merged prompt groups / 集約した質問 : "
                  f"{merge_stats['merged_prompt_groups']}")
            print(f"  input rows in those groups / 対象行 : "
                  f"{merge_stats['merged_input_rows']}")
            print(f"  rows after merge / 集約後の行数 : "
                  f"{merge_stats['rows_after_merge']}")
        else:
            info("No conflicting duplicate user prompts found. / "
                 "同じ質問に複数回答の行は見つかりませんでした。")

    good, stats, _ = normalise_and_validate(records, default_system=args.system)
    print_stats(stats, args.max_seq_length)
    if not good:
        return 1

    os.makedirs(args.out, exist_ok=True)

    # 1) ShareGPT for the Colab / Unsloth notebook
    sharegpt_path = os.path.join(args.out, "train_sharegpt.jsonl")
    write_jsonl(sharegpt_path, [to_sharegpt(m) for m in good])
    ok(f"wrote {sharegpt_path}  (Colab / Unsloth)")

    # 2) messages train/valid for MLX (local Mac)
    train, valid = split_train_valid(good, args.valid_frac, args.seed)
    train_path = os.path.join(args.out, "train.jsonl")
    valid_path = os.path.join(args.out, "valid.jsonl")
    write_jsonl(train_path, [{"messages": m} for m in train])
    write_jsonl(valid_path, [{"messages": m} for m in valid])
    ok(f"wrote {train_path} ({len(train)}) + {valid_path} ({len(valid)})  (MLX)")

    print()
    info("Next / 次は:")
    print("  Colab : upload " + sharegpt_path + " in the notebook's data step")
    print("          ノートブックのデータ手順で上のファイルをアップロード")
    print("  Mac   : python local/mlx_train.py --dataset " +
          os.path.join(args.out, "train.jsonl") + "   (see local/mac_mlx.md)")
    return 0


def cmd_validate(args):
    if not os.path.isfile(args.file):
        err(f"File not found / ファイルが見つかりません: {args.file}")
        return 2
    info(f"Validating {args.file} ...")
    try:
        records = load_records(args.file)
    except ValueError as e:
        err(str(e))
        return 1
    good, stats, _ = normalise_and_validate(records)
    print_stats(stats, args.max_seq_length)
    if stats["skipped"] or stats["count"] == 0:
        warn("Some rows had problems (see WARN lines above). / "
             "問題のある行があります(上の WARN を参照)。")
        return 1
    ok("File looks good. / ファイルは問題ありません。")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="prepare_data.py",
        description="Prepare / check training data for Gemma 4 E2B fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    b = sub.add_parser("build", help="convert a CSV/JSONL into training data")
    b.add_argument("input", help="path to your .csv / .jsonl / .json")
    b.add_argument("--system", default=None,
                   help="a personality/system instruction applied to rows without one")
    b.add_argument("--out", default="out", help="output folder (default: out)")
    b.add_argument("--valid-frac", type=float, default=0.1, dest="valid_frac",
                   help="fraction held out for validation (MLX), default 0.1")
    b.add_argument("--seed", type=int, default=3407, help="shuffle seed")
    b.add_argument("--max-seq-length", type=int, default=1024, dest="max_seq_length",
                   help="warn if an example is longer than this (default 1024)")
    b.add_argument("--merge-same-user", action="store_true",
                   help="merge rows with the exact same user prompt into one "
                        "multi-answer example / 同じ質問の複数回答を1例にまとめる")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="check an existing JSONL/CSV file")
    v.add_argument("file", help="path to the file to check")
    v.add_argument("--max-seq-length", type=int, default=1024, dest="max_seq_length")
    v.set_defaults(func=cmd_validate)
    return p


def main(argv):
    parser = build_parser()
    # Friendly shorthand: `prepare_data.py myfile.csv` == `build myfile.csv`
    # やさしい省略形: サブコマンドを省くと build とみなす
    if argv and argv[0] not in ("build", "validate", "-h", "--help"):
        argv = ["build"] + argv
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
