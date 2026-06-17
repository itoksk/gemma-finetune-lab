# Browser chat app / ブラウザのチャットアプリ

A single HTML file that chats with your fine-tuned model through Ollama's local
API. This is your model **embedded in a real app** — no build step, no
dependencies, nothing leaves your computer.

1つの HTML ファイルが、Ollama のローカル API を通じてあなたのモデルと会話します。
これがあなたのモデルを**アプリに組み込んだ**形です。ビルド不要・依存なし。

## Run it / 起動

First make sure your model exists in Ollama (see the repo `README.md` or
`local/import_to_ollama.sh`):

まず Ollama にモデルがあることを確認します:

```bash
ollama run my-gemma "hi"   # should reply / 返事が返ればOK
```

Then serve this folder and open it in a browser. Serving (rather than
double-clicking the file) avoids browser security blocks:

このフォルダをローカルサーバーで開きます(ダブルクリックより確実):

```bash
cd use/web-chat
python3 -m http.server
# open http://localhost:8000
```

Type the model's name in the **model** box (default `my-gemma`) and chat.
右上の **model** 欄にモデル名(初期値 `my-gemma`)を入れて会話します。

## If it can't connect / つながらないとき

The green/red dot top-left shows the Ollama connection. If it's red:

- Make sure Ollama is running. / Ollama が起動しているか確認。
- The model name must match what you created. / モデル名が一致しているか確認。
- Browser blocking the request? Allow the origin and restart Ollama:
  ブラウザにブロックされる場合はオリジンを許可して再起動:
  ```bash
  # macOS
  launchctl setenv OLLAMA_ORIGINS "*" && pkill Ollama; open -a Ollama
  ```

## Make it yours / カスタマイズ

It's one file — open `index.html` and edit. Ideas: change the title and colours,
add a "system" personality, or build it into your Day 4/5 hackathon project.

1ファイルなので `index.html` を開いて自由に編集できます。タイトルや色を変える、
性格(system)を足す、Day 4/5 のハッカソン作品に組み込む、など。
