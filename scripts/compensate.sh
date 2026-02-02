#!/bin/bash

# Configuration
input_dir="data/compensation"
output_dir="output/compensation"

setup_names=("cloud" "pillow" "velvet" "yellow_paint")
model_types=("brdf")

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

# Unified training loop
for setup_name in "${setup_names[@]}"; do
    for model_type in "${model_types[@]}"; do
        # Set model directory and extra arguments based on model type
        case "$model_type" in
            "brdf")
                model_dir="${output_dir}/brdf/setups/${setup_name}"
                extra_args=""
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
        python train.py -s "$input_dir/setups/$setup_name" -m "$model_dir"  --gpu_id "$gpu_id"

        # Stop GPU memory monitoring
        kill ${monitor_pid}

        # Run compensation for view IDs 26, 27, 28
        view_ids=(26 27 28)
        for view_id in "${view_ids[@]}"; do
            echo "Run compensate.py for setup: $setup_name, view_id: $view_id"
            python compensate.py --model_path "$model_dir" --root "$input_dir" --setup "$setup_name" --view_id "$view_id"
        done
    done
done
