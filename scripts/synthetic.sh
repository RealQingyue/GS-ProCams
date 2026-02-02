#!/bin/bash

# =============================================================================
# GS-ProCams Pipeline for Synthetic Datasets
# =============================================================================
# This script runs the complete pipeline: training, rendering, and evaluation
# For easy reproduction of the results
# =============================================================================

# Configuration
input_dir="data/nepmap-dataset"
output_dir="output/nepmap-dataset"

setup_names=("castle" "planck" "zoo" "pear")
model_types=("wo_psf")

gpu_id="0"

# Function to monitor GPU memory usage
monitor_gpu_memory() {
    # make directory if it doesn't exist
    mkdir -p "$1"
    local log_file="$1/gpu_memory.log"
    local gpu_id="$2"
    local start_time=$(date +%s)
    local counter=0
    
    echo "Starting GPU memory monitoring for GPU $gpu_id" >> "${log_file}"
    echo "Time, Runtime, GPU Memory(MB)" >> "${log_file}"
    
    while true; do
        local current_time=$(date +%s)
        local elapsed_seconds=$((current_time - start_time))
        local elapsed_minutes=$((elapsed_seconds / 60))
        local remaining_seconds=$((elapsed_seconds % 60))
        local memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id)
        
        counter=$((counter + 1))
        echo "$(date '+%H:%M:%S'), ${elapsed_minutes}min${remaining_seconds}s, ${memory_usage}MB" >> "${log_file}"
        
        sleep 10
    done
}

# Training loop
for setup_name in "${setup_names[@]}"; do
    for model_type in "${model_types[@]}"; do
        # Set model directory and extra arguments based on model type
        case "$model_type" in
            "wo_psf")
                model_dir="${output_dir}/wo_psf/setups/${setup_name}"
                extra_args="--wo_psf --white_background"
                ;;
            *)
                echo "Error: Unknown model type: $model_type"
                exit 1
                ;;
        esac

        # Start GPU memory monitoring in the background
        if [ -z "$model_dir" ]; then
            echo "Error: model_dir is empty"
            exit 1
        fi

        monitor_gpu_memory "$model_dir" "$gpu_id" &
        monitor_pid=$!

        # Run the training script
        echo "Run train.py ${extra_args} for: $setup_name"
        python train.py -s "$input_dir/$setup_name" -m "$model_dir" --evaluate ${extra_args} --gpu_id "$gpu_id"

        # Stop GPU memory monitoring
        kill ${monitor_pid}
    done
done
