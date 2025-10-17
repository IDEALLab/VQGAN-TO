#!/bin/bash
# Usage: bash pixel_swap_viz.sh <model_name> [num_swaps] [sample_idx]
# Example: bash pixel_swap_viz.sh 1_c1_online_64
# Example: bash pixel_swap_viz.sh 1_c1_online_64 16
# Example: bash pixel_swap_viz.sh 1_c1_online_64 16 5

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <model_name> [num_swaps] [sample_idx]"
    echo "  num_swaps defaults to 16 if not specified"
    echo "  sample_idx defaults to 1 if not specified"
    exit 1
fi

MODEL_NAME="$1"
NUM_SWAPS="${2:-64}"  # Default to 16 if second argument not provided
SAMPLE_IDX="${3:-5}"  # Default to 1 if third argument not provided
JOB_NAME="swap_${MODEL_NAME}_${NUM_SWAPS}_s${SAMPLE_IDX}"
WD="$HOME/scratch/VQGAN/src"
EVAL_DIR="$HOME/scratch/VQGAN/evals/$MODEL_NAME/latent_results"
SCRIPT_PATH="$WD/pixel_swap_viz.py"

mkdir -p "$EVAL_DIR"

sbatch <<EOL
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$HOME/scratch/slurm-report/slurm_swap_%j.out
#SBATCH -t 00:01:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue

. ~/.bashrc
source ~/vqgan_env/bin/activate
cd "$WD" || exit 1

echo "Running latent pixel swap for model: $MODEL_NAME with $NUM_SWAPS swaps on sample $SAMPLE_IDX"
nvidia-smi
env | grep SLURM

python "$SCRIPT_PATH" --model_name "$MODEL_NAME" --num_swaps "$NUM_SWAPS" --sample_idx "$SAMPLE_IDX" > "$EVAL_DIR/pixel_swap_${NUM_SWAPS}_sample_${SAMPLE_IDX}.txt" 2>&1
echo "Pixel swap completed at \$(date)" >> "$EVAL_DIR/pixel_swap_${NUM_SWAPS}_sample_${SAMPLE_IDX}.txt"
EOL
