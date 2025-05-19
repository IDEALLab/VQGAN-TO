#!/bin/bash
#SBATCH --job-name=train_transformer
#SBATCH --array=1-1
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_main-%A_%a.out
#SBATCH -t 12:00:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=verbose,per_task:1

. ~/.bashrc
runname=$(date +"Tr-%Y-%m-%d_%H-%M-%S")
wd=~/scratch/VQGAN/src
sd=~/scratch/VQGAN/saves/$runname
mkdir -p "$sd"

# Make sure to unload python to prevent conflict with default installed packages on HPC
module unload python
source ~/vqgan_env/bin/activate
cd "$wd"
python training_transformer.py --run_name "$runname" > "$sd/output.txt" 2>&1