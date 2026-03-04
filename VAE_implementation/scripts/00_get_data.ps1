# 00_get_data.ps1 — Clone or update the RF CSV dataset into data/raw/
# Run from repo root: .\VAE_implementation\scripts\00_get_data.ps1

$ErrorActionPreference = "Stop"

# Repo root = folder where this script lives: ...\VAE_implementation\scripts -> go up 2
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$RawDir = Join-Path $RootDir "data\raw"
$DatasetDir = Join-Path $RawDir "DataBase-RF-FM-88MHz-108MHz-Bogota-Funza"
$RepoUrl = "https://github.com/dramirezbe/DataBase-RF-FM-88MHz-108MHz-Bogota-Funza.git"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null

if (Test-Path (Join-Path $DatasetDir ".git")) {
    Write-Host "[OK] Dataset already exists: $DatasetDir"
    Write-Host "[INFO] Pulling latest changes..."
    git -C $DatasetDir pull
} else {
    Write-Host "[INFO] Cloning dataset into: $DatasetDir"
    git clone $RepoUrl $DatasetDir
}

Write-Host "[DONE] Raw dataset ready."
Write-Host "Path: $DatasetDir"