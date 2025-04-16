#!/bin/bash
#SBATCH --job-name=eval_vqgan
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_eval-%j.out
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 4:00:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --gpus=h100:1

# Check if model name is provided
if [ -z "$1" ]; then
    echo "Error: Model name must be provided as an argument"
    echo "Usage: sbatch eval.sh <model_name>"
    exit 1
fi

# Set model name from command line argument
model_name=$1
echo "Evaluating model: $model_name"

# Setup environment
. ~/.bashrc
wd=~/scratch/VQGAN/src
eval_dir=~/scratch/VQGAN/evals/$model_name
mkdir -p "$eval_dir"

# Make sure to unload python to prevent conflict with default installed packages on HPC
module unload python
source ~/vqgan_env/bin/activate
cd "$wd"

# Run evaluation
echo "Starting evaluation at $(date)"
python eval_vqgan.py --model-name "$model_name" > "$eval_dir/eval_output.txt" 2>&1

# Copy results to a timestamped directory if needed
# eval_time=$(date +"%Y-%m-%d_%H-%M-%S")
# mkdir -p "$eval_dir/runs/$eval_time"
# cp -r "$eval_dir/results" "$eval_dir/runs/$eval_time/"

echo "Evaluation completed at $(date)"