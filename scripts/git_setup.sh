#!/bin/bash
# ============================================================
# Git Setup Script für Football Predictor
# Einmalig ausführen nach dem Klonen / ersten Start
# Usage: bash scripts/git_setup.sh
# ============================================================

set -e

echo "⚽ Football Predictor — Git Setup"
echo "=================================="

# 1. Git initialisieren
git init
echo "✅ Git initialized"

# 2. Remote setzen (ersetze URL mit deinem GitHub Repo!)
# git remote add origin https://github.com/DEIN_USERNAME/football-predictor.git

# 3. Ersten Commit auf main
git add .
git commit -m "feat: initial project structure

- Folder structure (src, data, config, tests, scripts, dashboard)
- Database layer (SQLite via SQLAlchemy)
- Football-data.org API collector
- Config & environment management
- Logging setup (loguru)
- Unit tests (pytest)
- README, .gitignore, requirements.txt"

echo "✅ Initial commit on main"

# 4. Branch-Struktur anlegen
git checkout -b develop
echo "✅ Branch: develop"

git checkout -b feature/data-collector
echo "✅ Branch: feature/data-collector"

git checkout develop
git checkout -b feature/feature-engineering
echo "✅ Branch: feature/feature-engineering"

git checkout develop
git checkout -b feature/poisson-model
echo "✅ Branch: feature/poisson-model"

git checkout develop
git checkout -b feature/monte-carlo
echo "✅ Branch: feature/monte-carlo"

git checkout develop
git checkout -b feature/ml-model
echo "✅ Branch: feature/ml-model"

git checkout develop
git checkout -b feature/dashboard
echo "✅ Branch: feature/dashboard"

# 5. Zurück zu develop für die Arbeit
git checkout develop

echo ""
echo "🎉 Setup complete! Branch-Übersicht:"
git branch -a

echo ""
echo "👉 Nächste Schritte:"
echo "   1. GitHub Repo erstellen unter: https://github.com/new"
echo "   2. Remote setzen: git remote add origin https://github.com/USERNAME/football-predictor.git"
echo "   3. Pushen: git push -u origin main && git push --all"
echo "   4. cp config/.env.example config/.env → API Keys eintragen"
echo "   5. pip install -r requirements.txt"
echo "   6. python scripts/init_db.py"
