#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENV=expert-rag

# 1️⃣  Ensure Miniconda
if ! command -v conda &>/dev/null; then
  echo "🔧 Installing Miniconda (first run only)…"
  curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash miniconda.sh -b -p "$HOME/miniconda"
  eval "$($HOME/miniconda/bin/conda shell.bash hook)"
fi

# 2️⃣  Create env if missing
if ! conda env list | grep -q "$ENV"; then
  echo "📦 Creating env '$ENV'…"
  conda env create -f environment.yml
fi

# 3️⃣  Launch UI
echo "⏳ The application is now loading. There might be a delay before it proceeds..."
set +u  
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
python app/rag_ui.py
echo "✅ Application has finished."



