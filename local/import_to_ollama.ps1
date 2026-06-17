# import_to_ollama.ps1 — register your fine-tuned model in Ollama (Windows).
# ファインチューニングしたモデルを Ollama に登録します (Windows)。
#
# Usage / 使い方:
#   powershell -ExecutionPolicy Bypass -File local\import_to_ollama.ps1 my-gemma $HOME\Downloads
#
# The <folder> must contain a `Modelfile` and the file(s) it points to.
param(
  [Parameter(Mandatory = $true)][string]$Name,
  [string]$Dir = "."
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Error "Ollama is not installed. Get it at https://ollama.com/download"
  exit 1
}
if (-not (Test-Path (Join-Path $Dir "Modelfile"))) {
  Write-Error "No Modelfile found in: $Dir  /  $Dir に Modelfile がありません。"
  exit 1
}

# Make sure the Ollama server is up / サーバー起動確認
try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 2 | Out-Null }
catch {
  Write-Host "Starting Ollama... / Ollama を起動中…"
  Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
  for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 2 | Out-Null; break }
    catch { Start-Sleep -Seconds 1 }
  }
}

Write-Host "==> ollama create $Name -f $Dir\Modelfile"
Push-Location $Dir
try { ollama create $Name -f Modelfile } finally { Pop-Location }

Write-Host ""
Write-Host "Done! / 完了!  Try it / 試す:"
Write-Host "  ollama run $Name `"Hello!`""
Write-Host ""
Write-Host "Use it elsewhere / ほかでも使う:"
Write-Host "  - Browser chat app: open use\web-chat\index.html, set the model name to `"$Name`""
Write-Host "  - vibe-local:       see use\vibe-local.md"
