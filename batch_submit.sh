#!/bin/bash
# Script to submit multiple VQGAN/CVQGAN/Transformer training jobs with different configurations
# and then submit evaluation jobs after each training job completes

# Function to print usage information
function print_usage() {
    echo "Usage: bash batch_submit_with_transformer.sh [options]"
    echo ""
    echo "Options:"
    echo "  -c, --configs CONFIG_FILE    Path to configuration file (default: configs.txt)"
    echo "  -f, --force                  Force job submission even if directories exist"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Example configuration file format:"
    echo "# Each job is defined by JOB_NAME:param1=value1,param2=value2"
    echo "transformer_exp:is_t=true,n_layer=12,n_head=12,n_embd=768"
}

CONFIG_FILE="configs.txt"
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--configs)
            CONFIG_FILE="$2"
            shift 2
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

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

TEMP_DIR=$(mktemp -d)
echo "Created temporary directory for job scripts: $TEMP_DIR"

function is_dir_nonempty() {
    local dir="$1"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        return 0
    else
        return 1
    fi
}

LINE_NUM=0
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    ((LINE_NUM++))
    JOB_NAME=${line%%:*}
    PARAMS=${line#*:}
    RUNNAME="${JOB_NAME}"

    IS_TRANSFORMER=false
    IS_CONDITIONAL_VQGAN=false

    IFS=',' read -ra PARAM_PAIRS <<< "$PARAMS"
    for pair in "${PARAM_PAIRS[@]}"; do
        KEY=${pair%%=*}
        VALUE=${pair#*=}
        lower_value="${VALUE,,}"
        if [[ "$KEY" == "is_t" && ("$lower_value" == "true" || "$lower_value" == "1") ]]; then
            IS_TRANSFORMER=true
        elif [[ "$KEY" == "is_c" && ("$lower_value" == "true" || "$lower_value" == "1") ]]; then
            IS_CONDITIONAL_VQGAN=true
        fi
    done

    if [ "$IS_TRANSFORMER" = true ]; then
        PYTHON_SCRIPT="training_transformer.py"
        JOB_TYPE="transformer"
    else
        PYTHON_SCRIPT="training_vqgan.py"
        JOB_TYPE=$([ "$IS_CONDITIONAL_VQGAN" = true ] && echo "cvqgan" || echo "vqgan")
    fi

    JOB_SCRIPT="$TEMP_DIR/run_${JOB_NAME}.sh"
    EVAL_SCRIPT="$TEMP_DIR/eval_${JOB_NAME}.sh"
    SAVE_DIR="$HOME/scratch/VQGAN/saves/$RUNNAME"
    EVAL_DIR="$HOME/scratch/VQGAN/evals/$RUNNAME"

    SAVE_EXISTS=false
    EVAL_EXISTS=false

    is_dir_nonempty "$SAVE_DIR" && SAVE_EXISTS=true
    is_dir_nonempty "$EVAL_DIR" && EVAL_EXISTS=true

    if [ "$SAVE_EXISTS" = true ] && [ "$EVAL_EXISTS" = true ] && [ "$FORCE" = false ]; then
        echo "Skipping job '$JOB_NAME': Save and eval directories exist."
        continue
    fi

    PARAM_STRING=""
    for pair in "${PARAM_PAIRS[@]}"; do
        KEY=${pair%%=*}
        VALUE=${pair#*=}
        lower_value="${VALUE,,}"
        if [[ "$lower_value" == "true" ]]; then VALUE="True"; fi
        if [[ "$lower_value" == "false" ]]; then VALUE="False"; fi
        PARAM_STRING+=" --$KEY $VALUE"
    done

    cat > "$JOB_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/home/adrake17/scratch/slurm-report/slurm_${JOB_TYPE}-%A_%a.out
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
wd=~/scratch/VQGAN/src
sd=~/scratch/VQGAN/saves/\$runname

if [ -d "\$sd" ] && [ "\$(ls -A "\$sd")" ]; then
    if [ -f "\$sd/training_status.txt" ] && grep -q "Training completed successfully" "\$sd/training_status.txt"; then
        echo "Training already completed. Exiting."
        exit 0
    else
        [ -f "\$sd/output.txt" ] && mv "\$sd/output.txt" "\$sd/output.txt.bak.\$(date +%Y%m%d%H%M%S)"
    fi
else
    mkdir -p "\$sd"
fi

module unload python
source ~/vqgan_env/bin/activate
cd "\$wd"

echo "Running $PYTHON_SCRIPT with: $PARAM_STRING --run_name \$runname"
python $PYTHON_SCRIPT $PARAM_STRING --run_name "\$runname" > "\$sd/output.txt" 2>&1

if [ \$? -eq 0 ]; then
    echo "Training completed successfully at \$(date)" > "\$sd/training_status.txt"
    exit 0
else
    echo "Training failed at \$(date)" > "\$sd/training_status.txt"
    exit 1
fi
EOL

    chmod +x "$JOB_SCRIPT"

    if [ "$SAVE_EXISTS" = true ] && [ "$FORCE" = false ]; then
        echo "Skipping training job for '$JOB_NAME'"
    else
        echo "Submitting training job: sbatch $JOB_SCRIPT"
        TRAIN_JOBID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')

        if [ -n "$TRAIN_JOBID" ]; then
            echo "Submitted training job with ID: $TRAIN_JOBID"

            if [ "$EVAL_EXISTS" = true ] && [ "$FORCE" = false ]; then
                echo "Skipping evaluation job for '$JOB_NAME': Eval directory exists"
            else
                EVAL_PY_SCRIPT="eval_vqgan.py"
                EVAL_ARG="--model_name \$runname"
                [ "$IS_TRANSFORMER" = true ] && EVAL_PY_SCRIPT="eval_transformer.py" && EVAL_ARG="--t_name \$runname"

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

. ~/.bashrc
runname="$RUNNAME"
wd=~/scratch/VQGAN/src
eval_dir=~/scratch/VQGAN/evals/\$runname

module unload python
source ~/vqgan_env/bin/activate
cd "\$wd"

if [ -d "\$eval_dir" ] && [ -f "\$eval_dir/eval_output.txt" ] && grep -q "Evaluation completed" "\$eval_dir/eval_output.txt"; then
    echo "Evaluation already completed."
    exit 0
fi

mkdir -p "\$eval_dir"
python $EVAL_PY_SCRIPT $EVAL_ARG > "\$eval_dir/eval_output.txt" 2>&1
echo "Evaluation completed at \$(date)" >> "\$eval_dir/eval_output.txt"
EOL
                chmod +x "$EVAL_SCRIPT"
                echo "Submitting evaluation job with dependency on $TRAIN_JOBID"
                sbatch --dependency=afterok:$TRAIN_JOBID "$EVAL_SCRIPT"
            fi
        else
            echo "Training job failed to submit for '$JOB_NAME'"
        fi
    fi
done < "$CONFIG_FILE"

echo "All jobs submitted. Scripts in: $TEMP_DIR"
