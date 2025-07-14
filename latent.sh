#!/bin/bash
# Usage: bash submit_latent_analysis.sh <model_name>
# Example: bash submit_latent_analysis.sh vq_gan_Feb10_15-57-05

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME="$1"
JOB_NAME="latent_${MODEL_NAME}"
WD="$HOME/scratch/VQGAN/src"
EVAL_DIR="$HOME/scratch/VQGAN/evals/$MODEL_NAME/latent_results"
SCRIPT_PATH="$WD/latent_analysis.py"

mkdir -p "$EVAL_DIR"

sbatch <<EOL
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$HOME/scratch/slurm-report/slurm_latent_%j.out
#SBATCH -t 02:00:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue

. ~/.bashrc
source ~/vqgan_env/bin/activate
cd "$WD" || exit 1

echo "Running latent analysis for model: $MODEL_NAME"
nvidia-smi
env | grep SLURM

python "$SCRIPT_PATH" --model_name "$MODEL_NAME" > "$EVAL_DIR/latent_output.txt" 2>&1
echo "Latent analysis completed at \$(date)" >> "$EVAL_DIR/latent_output.txt"
EOL