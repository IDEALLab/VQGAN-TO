#!/bin/bash
# Script to submit multiple VQGAN training jobs with different configurations
# and then submit evaluation jobs after each training job completes
# Modified to ensure exclusive GPU and CPU resource allocation on Zaratan

# Function to print usage information
function print_usage() {
    echo "Usage: bash batch_submit.sh [options]"
    echo ""
    echo "Options:"
    echo "  -c, --configs CONFIG_FILE    Path to configuration file (default: configs.txt)"
    echo "  -d, --dry-run                Print commands without submitting jobs"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Example configuration file format:"
    echo "# Comments start with #"
    echo "# Each job is defined by JOB_NAME:param1=value1,param2=value2"
    echo "experiment1:batch_size=64,learning_rate=0.001,epochs=100"
    echo "experiment2:batch_size=128,learning_rate=0.0005,epochs=150"
}

# Parse command-line arguments
CONFIG_FILE="configs.txt"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--configs)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Create a temp directory for job scripts
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory for job scripts: $TEMP_DIR"

# Process each configuration line
LINE_NUM=0
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Increment line number for job naming
    ((LINE_NUM++))
    
    # Split the line into job name and parameters
    JOB_NAME=${line%%:*}
    PARAMS=${line#*:}
    
    # Create a custom run.sh script for this job
    JOB_SCRIPT="$TEMP_DIR/run_${JOB_NAME}.sh"
    EVAL_SCRIPT="$TEMP_DIR/eval_${JOB_NAME}.sh"
    
    # Use the given name for this run
    RUNNAME="${JOB_NAME}"
    
    # Prepare the parameter string for the Python command
    PARAM_STRING=""
    IFS=',' read -ra PARAM_PAIRS <<< "$PARAMS"
    for pair in "${PARAM_PAIRS[@]}"; do
        # Skip empty parameters (for baseline case)
        if [[ -z "$pair" ]]; then
            continue
        fi
        
        KEY=${pair%%=*}
        VALUE=${pair#*=}
        
        # Handle space-separated lists (like decoder_channels)
        if [[ "$VALUE" == *" "* ]]; then
            # For space-separated values, we need to pass each value separately
            PARAM_STRING="$PARAM_STRING --$KEY $VALUE"
        else
            PARAM_STRING="$PARAM_STRING --$KEY $VALUE"
        fi
    done
    
    # Create the training job script with node exclusivity
    cat > "$JOB_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=1-1
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_main-%A_%a.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --nodes=1-1
#SBATCH --mail-user=adrake17@umd.edu
#SBATCH --mail-type=END

. ~/.bashrc
runname="$RUNNAME"
echo "Starting job with name: \$runname"
wd=~/scratch/VQGAN/src
sd=~/scratch/VQGAN/saves/\$runname
mkdir -p "\$sd"

# Print resource allocation information
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Copy this job script to the save directory for reference
cp "$JOB_SCRIPT" "\$sd/job_script.sh"

# Make sure to unload python to prevent conflict with default installed packages on HPC
module unload python
source ~/vqgan_env/bin/activate
cd "\$wd"

# Run training with the specified parameters
echo "Running with parameters: $PARAM_STRING --run-name \$runname"
python training_vqgan.py $PARAM_STRING --run-name "\$runname" > "\$sd/output.txt" 2>&1

# Write success/failure status to a file for reference
if [ \$? -eq 0 ]; then
    echo "Training completed successfully at \$(date)" > "\$sd/training_status.txt"
    exit 0
else
    echo "Training failed with exit code \$? at \$(date)" > "\$sd/training_status.txt"
    exit 1
fi
EOL

    # Create the evaluation job script with exclusive resource allocation
    cat > "$EVAL_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=eval_${JOB_NAME}
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_eval-%j.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --exclusive=user
#SBATCH -t 0:30:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH --gpu-bind=verbose,per_task:1

# Model name is passed from the batch submit script
model_name="$RUNNAME"
echo "Evaluating model: \$model_name"

# Print resource allocation information
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Setup environment
. ~/.bashrc
wd=~/scratch/VQGAN/src
eval_dir=~/scratch/VQGAN/evals/\$model_name
mkdir -p "\$eval_dir"

# Make sure to unload python to prevent conflict with default installed packages on HPC
module unload python
source ~/vqgan_env/bin/activate
cd "\$wd"

# Run evaluation
echo "Starting evaluation at \$(date)"
python eval_vqgan.py --model-name "\$model_name" > "\$eval_dir/eval_output.txt" 2>&1

echo "Evaluation completed at \$(date)"
EOL

    # Make both scripts executable
    chmod +x "$JOB_SCRIPT"
    chmod +x "$EVAL_SCRIPT"
    
    # Submit or print the job command
    if [ "$DRY_RUN" = true ]; then
        echo "Would submit training job: sbatch $JOB_SCRIPT"
        echo "Would submit evaluation job: sbatch --dependency=afterok:\$JOBID $EVAL_SCRIPT"
    else
        echo "Submitting training job: sbatch $JOB_SCRIPT"
        TRAIN_JOBID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
        
        if [ -n "$TRAIN_JOBID" ]; then
            echo "Submitted training job $JOB_NAME with ID: $TRAIN_JOBID"
            
            # Submit evaluation job with dependency on training job
            echo "Submitting evaluation job with dependency on job $TRAIN_JOBID"
            EVAL_JOBID=$(sbatch --dependency=afterok:$TRAIN_JOBID "$EVAL_SCRIPT" | awk '{print $4}')
            echo "Submitted evaluation job for $JOB_NAME with ID: $EVAL_JOBID"
        else
            echo "Error: Failed to submit training job $JOB_NAME"
        fi
    fi
    
done < "$CONFIG_FILE"

echo "Job submission process completed."
echo "Temporary job scripts are stored in: $TEMP_DIR"
echo "You may delete this directory when all jobs have completed."