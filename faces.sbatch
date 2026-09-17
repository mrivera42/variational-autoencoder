#!/bin/bash
#SBATCH -J faces-ddpm
#SBATCH --account=gts-awu36
#SBATCH -N1
#SBATCH --gres=gpu:1                 # ONE GPU is plenty for sampling (any type)
#SBATCH --mem-per-gpu=40G
#SBATCH -t1:00:00                    # sampling takes ~10 min; 1h is a safe ceiling
#SBATCH -q embers
#SBATCH -o logs/faces_%j.out

# --- EDIT THIS to the dir where you put the two files on PACE ---
WORKDIR=$HOME/p-awu36-0/faces-ddpm
cd "$WORKDIR"

# HF cache must be pre-populated on the LOGIN node (compute nodes have no internet).
export HF_HOME="$WORKDIR/hf_cache"
export HF_HUB_OFFLINE=1              # use the pre-downloaded checkpoint, no network

mkdir -p logs

# Uses the uv env you built on the login node (offline: no package downloads here).
uv run python faces_noise_ablation.py
