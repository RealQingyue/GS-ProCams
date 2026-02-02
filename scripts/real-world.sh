#!/bin/bash

# =============================================================================
# GS-ProCams Pipeline for Real-World Datasets
# =============================================================================
# This script runs the complete pipeline: training, rendering, and evaluation
# For easy reproduction of the results
# =============================================================================

# Configuration
input_dir="data/real-world"
output_dir="output/real-world"

setup_names=("basketball" "bottles" "coffee" "chikawa" "color" "projector" "wukong")
model_types=("brdf")
num_views=(25) # Ablation 1: (20, 10, 5, 4, 2) for viewpoints number ablation
num_images=()   # Ablation 2: (75, 50, 25) for images number ablation

gpu_id="0"

# Pipeline control flags
run_training=true
run_rendering=true
run_evaluation=true
skip_existing=false

# =============================================================================
# Helper Functions
# =============================================================================

# Simple logging function
log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Function to check if training is complete
is_training_complete() {
    local model_dir="$1"
    # Check for the final checkpoint or completion marker
    [[ -f "$model_dir/procams/iteration_20000/procams.ckpt" ]] || [[ -f "$model_dir/point_cloud/iteration_20000/point_cloud.ply" ]]
}

# Function to check if rendering is complete
is_rendering_complete() {
    local save_dir="$1"
    # Check for render completion marker or key output files
    [[ -d "$save_dir" ]] && [[ $(find "$save_dir" -name "*.png" | wc -l) -gt 0 ]]
}

# =============================================================================
# Training Function
# =============================================================================
run_training() {
    local setup_name=$1
    local model_type=$2
    local param_name=$3
    local param_value=$4

    # Set model directory and extra arguments
    case "$model_type" in
        "brdf")
            model_dir="${output_dir}/brdf/setups/${setup_name}/${param_name}/${param_value}"
            extra_args="--${param_name} ${param_value}"
            ;;
        "wo_psf")
            model_dir="${output_dir}/wo_psf/setups/${setup_name}/${param_name}/${param_value}"
            extra_args="--wo_psf --${param_name} ${param_value}"
            ;;
        *)
            log "Unknown model type: $model_type"
            exit 1
            ;;
    esac

    # Check if training should be skipped
    if [[ "$skip_existing" == "true" ]] && is_training_complete "$model_dir"; then
        return 0
    fi

    # Run the training script
    if python train.py -s "$input_dir/setups/$setup_name" -m "$model_dir" ${extra_args} --gpu_id "$gpu_id"; then
        return 0
    else
        log "Training failed: $setup_name-$model_type-${param_name}_${param_value}"
        return 1
    fi
}

# =============================================================================
# Rendering Function
# =============================================================================
run_render() {
    local setup_name=$1
    local model_dir=$2
    local save_dir=$3

    # Check if rendering should be skipped
    if [[ "$skip_existing" == "true" ]] && is_rendering_complete "$save_dir"; then
        return 0
    fi

    # Check if trained model exists
    if ! is_training_complete "$model_dir"; then
        return 1
    fi

    # Run the render.py script
    if python render.py -r "$input_dir" -s "$setup_name" -m "$model_dir" -o "$save_dir" --test_fps --render_scene --gpu_id "$gpu_id"; then
        return 0
    else
        log "Rendering failed: $setup_name"
        return 1
    fi
}

# =============================================================================
# Training and Rendering Loop
# =============================================================================

# Main loop
for setup_name in "${setup_names[@]}"; do
    for model_type in "${model_types[@]}"; do
        
        # Process num_view configurations
        for num_view in "${num_views[@]}"; do
            model_dir="${output_dir}/${model_type}/setups/${setup_name}/num_view/${num_view}"
            save_dir="${model_dir}/render"
            
            # Training phase
            if [[ "$run_training" == "true" ]]; then
                run_training "$setup_name" "$model_type" "num_view" "$num_view"
            fi
            
            # Rendering phase
            if [[ "$run_rendering" == "true" ]]; then
                run_render "$setup_name" "$model_dir" "$save_dir"
            fi
        done
        
        # Process num_images configurations
        for num_image in "${num_images[@]}"; do
            model_dir="${output_dir}/${model_type}/setups/${setup_name}/num_images/${num_image}"
            save_dir="${model_dir}/render"
            
            # Training phase
            if [[ "$run_training" == "true" ]]; then
                run_training "$setup_name" "$model_type" "num_images" "$num_image"
            fi
            
            # Rendering phase
            if [[ "$run_rendering" == "true" ]]; then
                run_render "$setup_name" "$model_dir" "$save_dir"
            fi
        done
    done
done

# =============================================================================
# Evaluation
# =============================================================================

if [[ "$run_evaluation" == "true" ]]; then

    # Build evaluation command arguments
    eval_args=()
    eval_args+=("--models_dir" "$output_dir/brdf")
    eval_args+=("--gt_dir" "$input_dir")
    eval_args+=("--output_dir" "$output_dir/evaluation")
    
    # Only add num_view_list if array is not empty
    if [[ ${#num_views[@]} -gt 0 ]]; then
        eval_args+=("--num_view_list" "${num_views[@]}")
    fi
    
    # Only add num_images_list if array is not empty
    if [[ ${#num_images[@]} -gt 0 ]]; then
        eval_args+=("--num_images_list" "${num_images[@]}")
    fi

    # Run evaluation
    python evaluate.py "${eval_args[@]}"
    eval_status=$?

    if [[ $eval_status -eq 0 ]]; then
        log "Evaluation completed successfully!"
    else
        log "Evaluation failed!"
        exit 1
    fi
fi
