#!/bin/bash
#SBATCH --job-name=batch_h1-nvidia        # Job name
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --ntasks-per-node=1               # One task per node
#SBATCH --gres=gpu:1                      # 1 GPU per job
#SBATCH --cpus-per-task=10                # 10 CPU cores per task
#SBATCH --hint=nomultithread              # Physical cores only
#SBATCH --time=5:00:00                    # Maximum execution time (HH:MM:SS). Maximum 20h
#SBATCH --output=logs/out/%x_%A_%a.out        # Output log
#SBATCH --error=logs/err/%x_%A_%a.err         # Error log
#SBATCH --array=0-4                       # Create an array of jobs. Will affect the value of $SLURM_ARRAY_TASK_ID

# Activate python venv
source ${ALL_CCFRWORK}/envIsaac/bin/activate

# Define seeds
declare -A CONFIGURATIONS=(
    ["0"]="agent.seed=0"
    ["1"]="agent.seed=10"
    ["2"]="agent.seed=20"
    ["3"]="agent.seed=30"
    ["4"]="agent.seed=40"
)

# Get current configuration
CONFIG="${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID]}"

# Create experiment name with array ID for uniqueness
EXPERIMENT_NAME="${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}"

# Run training
set -x
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-H1-v0 \
    --headless \
    --num_envs=4096 \
    agent.max_iterations=1000 \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    ${CONFIG}

echo "Job ${SLURM_ARRAY_TASK_ID} completed successfully"
