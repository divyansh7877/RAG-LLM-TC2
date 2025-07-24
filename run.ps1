$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root
$envName = "expert-rag"

# 1️⃣  Miniconda check
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Host "🔧 Installing Miniconda (first run only)…"
  $url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
  $exe = "$env:TEMP\miniconda.exe"
  Invoke-WebRequest $url -OutFile $exe
  Start-Process -Wait $exe -ArgumentList "/S","/D=$env:USERPROFILE\miniconda"
  & "$env:USERPROFILE\miniconda\Scripts\conda.exe" init powershell | Out-Null
  $env:Path += ";$env:USERPROFILE\miniconda;$env:USERPROFILE\miniconda\Scripts"
}

# 2️⃣  Conda env
if (-not ((conda env list) -match $envName)) {
  Write-Host "📦 Creating env '$envName'…"
  conda env create -f environment.yml
}

# 3️⃣  Launch
Write-Host "🚀 Starting Expert-Call RAG UI…"
conda run -n $envName python app\rag_ui.py
