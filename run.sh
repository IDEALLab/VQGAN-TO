#!/bin/bash
#SBATCH --job-name=topopt
#SBATCH --array=1-1
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_main-%A_%a.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --mail-user=adrake17@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# === Load environment ===
. ~/.bashrc

# === Define paths ===
runname=$(date +"%Y-%m-%d_%H-%M-%S")
wd=~/scratch/VQGAN/src
sd=~/scratch/VQGAN/saves/$runname

mkdir -p "$sd"

# === Load modules ===
module unload python

# === Activate virtual environment ===
source ~/vqgan_env/bin/activate || {
    echo "❌ Failed to activate virtualenv"
    exit 1
}

# === Go to working directory ===
cd "$wd" || {
    echo "❌ Failed to cd to $wd"
    exit 1
}

# === Run training ===
echo "✅ Starting training at $(date)" | tee "$sd/runlog.txt"
python training_vqgan.py > "$sd/output.txt" 2>&1
echo "✅ Finished training at $(date)" >> "$sd/runlog.txt"