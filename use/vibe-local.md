# Use your fine-tune inside vibe-local / vibe-local で自分のモデルを使う

[vibe-local](https://github.com/itoksk/vibe-local-private) is the course's local
AI coding partner in the terminal. Once your fine-tuned model is in Ollama, you
can make vibe-local talk through *your* model.

vibe-local はターミナルで動く、このコースのローカル AI 相棒です。あなたのモデルを
Ollama に取り込めば、vibe-local を*あなたのモデル*で動かせます。

## 1. Make sure the model is in Ollama / モデルを Ollama に登録

```bash
ollama list           # is "my-gemma" listed? / 一覧に my-gemma はある？
# if not, import it first:
bash local/import_to_ollama.sh my-gemma ~/Downloads
```

vibe-local auto-discovers any model installed in Ollama, so this is the only
prerequisite. vibe-local は Ollama にあるモデルを自動で見つけます。

## 2. Point vibe-local at your model / vibe-local に指定する

**Easiest — just for one run / 一回だけ:**

```bash
vibe-local --model my-gemma
```

**Make it the default / 既定にする** — edit `~/.config/vibe-local/config` and set:

```bash
MODEL="my-gemma"
```

(Or set the environment variable `VIBE_LOCAL_MODEL=my-gemma` before launching.)

## 3. Chat / 会話する

```bash
vibe-local
```

Now your custom personality answers in the terminal. 端末であなたの個性が応答します。

## Notes / メモ

- A 2B model like Gemma 4 E2B is great for chat and small tasks, but it is **not
  a strong coding model** — for real coding in vibe-local, keep using `qwen3:8b`
  and switch to `my-gemma` when you want your custom character.
  Gemma 4 E2B はチャットや小さな作業には良いですが、**本格的なコーディングには
  非力**です。コーディングは `qwen3:8b`、自作キャラを使いたいときに `my-gemma`。
- If replies look off, re-export at higher quality (`q8_0`) from the notebook —
  see the repo `README.md` troubleshooting.
