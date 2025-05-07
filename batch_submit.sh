#!/bin/bash
# Script to submit multiple VQGAN training jobs with different configurations
# and then submit evaluation jobs after each training job completes

# Function to print usage information
function print_usage() {
    echo "Usage: bash batch_submit.sh [options]"
    echo ""
    echo "Options:"
    echo "  -c, --configs CONFIG_FILE    Path to configuration file (default: configs.txt)"
    echo "  -d, --dry-run                Print commands without submitting jobs"
    echo "  -f, --force                  Force job submission even if directories exist"
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
FORCE=false

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
        -f|--force)
            FORCE=true
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

# Function to check if directory exists and is non-empty
function is_dir_nonempty() {
    local dir="$1"
    # Check if directory exists and has files in it
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        return 0  # True - directory exists and is non-empty
    else
        return 1  # False - directory doesn't exist or is empty
    fi
}

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
    
    # Check if save and eval directories already exist and are non-empty
    SAVE_DIR="$HOME/scratch/VQGAN/saves/$RUNNAME"
    EVAL_DIR="$HOME/scratch/VQGAN/evals/$RUNNAME"
    
    SAVE_EXISTS=false
    EVAL_EXISTS=false
    
    if is_dir_nonempty "$SAVE_DIR"; then
        SAVE_EXISTS=true
    fi
    
    if is_dir_nonempty "$EVAL_DIR"; then
        EVAL_EXISTS=true
    fi
    
    # Skip this job if directories exist and we're not forcing a rerun
    if [ "$SAVE_EXISTS" = true ] && [ "$EVAL_EXISTS" = true ] && [ "$FORCE" = false ]; then
        echo "Skipping job '$JOB_NAME': Save directory and eval directory already exist."
        echo "Use --force to resubmit this job anyway."
        continue
    fi
    
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
    
    # Create the training job script
    cat > "$JOB_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
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
runname="$RUNNAME"
echo "Starting job with name: \$runname"
wd=~/scratch/VQGAN/src
sd=~/scratch/VQGAN/saves/\$runname

# Check if directory already exists and has content
if [ -d "\$sd" ] && [ "\$(ls -A "\$sd" 2>/dev/null)" ]; then
    echo "Save directory \$sd already exists and contains files."
    echo "Checking if training completed successfully..."
    
    if [ -f "\$sd/training_status.txt" ] && grep -q "Training completed successfully" "\$sd/training_status.txt"; then
        echo "Training was already completed successfully. Exiting job."
        exit 0
    else
        echo "Previous training may not have completed successfully. Continuing with job."
        # Optionally make a backup of previous run
        if [ -f "\$sd/output.txt" ]; then
            mv "\$sd/output.txt" "\$sd/output.txt.bak.\$(date +%Y%m%d%H%M%S)"
        fi
    fi
else
    mkdir -p "\$sd"
fi

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
echo "Running with parameters: $PARAM_STRING --run_name \$runname"
python training_vqgan.py $PARAM_STRING --run_name "\$runname" > "\$sd/output.txt" 2>&1

# Write success/failure status to a file for reference
if [ \$? -eq 0 ]; then
    echo "Training completed successfully at \$(date)" > "\$sd/training_status.txt"
    exit 0
else
    echo "Training failed with exit code \$? at \$(date)" > "\$sd/training_status.txt"
    exit 1
fi
EOL

    # Create the evaluation job script
    cat > "$EVAL_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=eval_${JOB_NAME}
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_eval-%j.out
#SBATCH -t 0:30:00
#SBATCH -A fuge-prj-jrl
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
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

# Check if directory already exists and has content
if [ -d "\$eval_dir" ] && [ "\$(ls -A "\$eval_dir" 2>/dev/null)" ]; then
    echo "Evaluation directory \$eval_dir already exists and contains files."
    echo "Checking if evaluation completed successfully..."
    
    if [ -f "\$eval_dir/eval_output.txt" ] && grep -q "Evaluation completed" "\$eval_dir/eval_output.txt"; then
        echo "Evaluation was already completed. Exiting job."
        exit 0
    else
        echo "Previous evaluation may not have completed successfully. Continuing with job."
        # Optionally make a backup of previous evaluation
        if [ -f "\$eval_dir/eval_output.txt" ]; then
            mv "\$eval_dir/eval_output.txt" "\$eval_dir/eval_output.txt.bak.\$(date +%Y%m%d%H%M%S)"
        fi
    fi
else
    mkdir -p "\$eval_dir"
fi

# Make sure to unload python to prevent conflict with default installed packages on HPC
module unload python
source ~/vqgan_env/bin/activate
cd "\$wd"

# Run evaluation
echo "Starting evaluation at \$(date)"
python eval_vqgan.py --model-name "\$model_name" > "\$eval_dir/eval_output.txt" 2>&1

echo "Evaluation completed at \$(date)" >> "\$eval_dir/eval_output.txt"
EOL

    # Make both scripts executable
    chmod +x "$JOB_SCRIPT"
    chmod +x "$EVAL_SCRIPT"
    
    # Submit or print the job command
    if [ "$DRY_RUN" = true ]; then
        if [ "$SAVE_EXISTS" = true ] && [ "$FORCE" = false ]; then
            echo "[DRY RUN] Would SKIP training job for '$JOB_NAME': Directory already exists"
        else
            echo "[DRY RUN] Would submit training job: sbatch $JOB_SCRIPT"
        fi
        
        if [ "$EVAL_EXISTS" = true ] && [ "$FORCE" = false ]; then
            echo "[DRY RUN] Would SKIP evaluation job for '$JOB_NAME': Directory already exists"
        else
            echo "[DRY RUN] Would submit evaluation job: sbatch --dependency=afterok:\$JOBID $EVAL_SCRIPT"
        fi
    else
        # Only submit if we're forcing or the directory doesn't exist yet
        if [ "$SAVE_EXISTS" = true ] && [ "$FORCE" = false ]; then
            echo "Skipping training job submission for '$JOB_NAME': Save directory already exists"
            
            # Check if we should still run the evaluation job
            if [ "$EVAL_EXISTS" = false ] || [ "$FORCE" = true ]; then
                echo "Submitting just the evaluation job for '$JOB_NAME'"
                EVAL_JOBID=$(sbatch "$EVAL_SCRIPT" | awk '{print $4}')
                echo "Submitted evaluation job for $JOB_NAME with ID: $EVAL_JOBID"
            else
                echo "Skipping evaluation job submission for '$JOB_NAME': Eval directory already exists"
            fi
        else
            echo "Submitting training job: sbatch $JOB_SCRIPT"
            TRAIN_JOBID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
            
            if [ -n "$TRAIN_JOBID" ]; then
                echo "Submitted training job $JOB_NAME with ID: $TRAIN_JOBID"
                
                # Skip evaluation job if it already exists and we're not forcing
                if [ "$EVAL_EXISTS" = true ] && [ "$FORCE" = false ]; then
                    echo "Skipping evaluation job submission for '$JOB_NAME': Eval directory already exists"
                else
                    # Submit evaluation job with dependency on training job
                    echo "Submitting evaluation job with dependency on job $TRAIN_JOBID"
                    EVAL_JOBID=$(sbatch --dependency=afterok:$TRAIN_JOBID "$EVAL_SCRIPT" | awk '{print $4}')
                    echo "Submitted evaluation job for $JOB_NAME with ID: $EVAL_JOBID"
                fi
            else
                echo "Error: Failed to submit training job $JOB_NAME"
            fi
        fi
    fi
    
done < "$CONFIG_FILE"

echo "Job submission process completed."
echo "Temporary job scripts are stored in: $TEMP_DIR"
echo "You may delete this directory when all jobs have completed."