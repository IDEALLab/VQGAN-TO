#!/bin/bash
#SBATCH --job-name=train_vqgan
#SBATCH --array=1-1
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_main-%A_%a.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --mail-user=adrake17@umd.edu
#SBATCH --mail-type=END

. ~/.bashrc
runname=$(date +"%Y-%m-%d_%H-%M-%S")
wd=~/scratch/VQGAN/src
sd=~/scratch/VQGAN/saves/$runname
mkdir -p "$sd"

# Make sure to unload python to prevent conflict with default installed packages on HPC
module unload python
source ~/vqgan_env/bin/activate
cd "$wd"
python training_vqgan.py --run-name "$runname" > "$sd/output.txt" 2>&1