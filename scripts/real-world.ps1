# =============================================================================
# GS-ProCams Pipeline for Real-World Datasets - PowerShell Version
# =============================================================================
# This script runs the complete pipeline: training, rendering, and evaluation
# For easy reproduction of the results
# =============================================================================

# Configuration
$input_dir = if ($env:INPUT_DIR) { $env:INPUT_DIR } else { "data/real-world" }
$output_dir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "output/real-world" }

$setup_names = @("basketball", "bottles", "coffee", "chikawa", "color", "projector", "wukong")
$model_types = @("brdf")
$num_views = @(25) # Ablation 1: (20, 10, 5, 4, 2) for viewpoints number ablation
$num_images = @()   # Ablation 2: (75, 50, 25) for images number ablation

$gpu_id = if ($env:GPU_ID) { $env:GPU_ID } else { "0" }

# Pipeline control flags
$run_training = if ($env:RUN_TRAINING -eq "true") { $true } else { $false }
$run_rendering = if ($env:RUN_RENDERING -eq "true") { $true } else { $false }
$run_evaluation = if ($env:RUN_EVALUATION -eq "false") { $false } else { $true }
$skip_existing = if ($env:SKIP_EXISTING -eq "true") { $true } else { $false }

# =============================================================================
# Helper Functions
# =============================================================================

# Simple logging function
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message"
}

# Function to check if training is complete
function Test-TrainingComplete {
    param([string]$ModelDir)
    $procams_ckpt = Join-Path $ModelDir "procams\iteration_20000\procams.ckpt"
    $point_cloud_ply = Join-Path $ModelDir "point_cloud\iteration_20000\point_cloud.ply"
    
    return (Test-Path $procams_ckpt) -or (Test-Path $point_cloud_ply)
}

# Function to check if rendering is complete
function Test-RenderingComplete {
    param([string]$SaveDir)
    
    if (!(Test-Path $SaveDir)) {
        return $false
    }
    
    $png_files = Get-ChildItem -Path $SaveDir -Filter "*.png" -Recurse
    return $png_files.Count -gt 0
}

# =============================================================================
# Training Function
# =============================================================================
function Invoke-Training {
    param(
        [string]$SetupName,
        [string]$ModelType,
        [string]$ParamName,
        [string]$ParamValue
    )

    # Set model directory and extra arguments
    switch ($ModelType) {
        "brdf" {
            $model_dir = Join-Path $output_dir "brdf\setups\$SetupName\$ParamName\$ParamValue"
            $extra_args = @("--$ParamName", $ParamValue)
        }
        "wo_psf" {
            $model_dir = Join-Path $output_dir "wo_psf\setups\$SetupName\$ParamName\$ParamValue"
            $extra_args = @("--wo_psf", "--$ParamName", $ParamValue)
        }
        default {
            Write-Log "Unknown model type: $ModelType"
            exit 1
        }
    }

    # Check if training should be skipped
    if ($skip_existing -and (Test-TrainingComplete $model_dir)) {
        return $true
    }

    # Run the training script
    $input_path = Join-Path $input_dir "setups\$SetupName"
    $arguments = @("train.py", "-s", $input_path, "-m", $model_dir) + $extra_args + @("--gpu_id", $gpu_id)
    
    $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        return $true
    } else {
        Write-Log "Training failed: $SetupName-$ModelType-${ParamName}_$ParamValue"
        return $false
    }
}

# =============================================================================
# Rendering Function
# =============================================================================
function Invoke-Render {
    param(
        [string]$SetupName,
        [string]$ModelDir,
        [string]$SaveDir
    )

    # Check if rendering should be skipped
    if ($skip_existing -and (Test-RenderingComplete $SaveDir)) {
        return $true
    }

    # Check if trained model exists
    if (!(Test-TrainingComplete $ModelDir)) {
        return $false
    }

    # Run the render.py script
    $arguments = @("render.py", "-r", $input_dir, "-s", $SetupName, "-m", $ModelDir, "-o", $SaveDir, "--test_fps", "--render_scene", "--gpu_id", $gpu_id)
    
    $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        return $true
    } else {
        Write-Log "Rendering failed: $SetupName"
        return $false
    }
}

# =============================================================================
# Training and Rendering Loop
# =============================================================================

# Main loop
foreach ($setup_name in $setup_names) {
    foreach ($model_type in $model_types) {
        
        # Process num_view configurations
        foreach ($num_view in $num_views) {
            $model_dir = Join-Path $output_dir "$model_type\setups\$setup_name\num_view\$num_view"
            $save_dir = Join-Path $model_dir "render"
            
            # Training phase
            if ($run_training) {
                Invoke-Training -SetupName $setup_name -ModelType $model_type -ParamName "num_view" -ParamValue $num_view
            }
            
            # Rendering phase
            if ($run_rendering) {
                Invoke-Render -SetupName $setup_name -ModelDir $model_dir -SaveDir $save_dir
            }
        }
        
        # Process num_images configurations
        foreach ($num_image in $num_images) {
            $model_dir = Join-Path $output_dir "$model_type\setups\$setup_name\num_images\$num_image"
            $save_dir = Join-Path $model_dir "render"
            
            # Training phase
            if ($run_training) {
                Invoke-Training -SetupName $setup_name -ModelType $model_type -ParamName "num_images" -ParamValue $num_image
            }
            
            # Rendering phase
            if ($run_rendering) {
                Invoke-Render -SetupName $setup_name -ModelDir $model_dir -SaveDir $save_dir
            }
        }
    }
}

# =============================================================================
# Evaluation
# =============================================================================

if ($run_evaluation) {
    # Build evaluation command arguments
    $eval_args = @()
    $eval_args += @("--models_dir", (Join-Path $output_dir "brdf"))
    $eval_args += @("--gt_dir", $input_dir)
    $eval_args += @("--output_dir", (Join-Path $output_dir "evaluation"))
    
    # Only add num_view_list if array is not empty
    if ($num_views.Count -gt 0) {
        $eval_args += @("--num_view_list") + $num_views
    }
    
    # Only add num_images_list if array is not empty
    if ($num_images.Count -gt 0) {
        $eval_args += @("--num_images_list") + $num_images
    }

    # Run evaluation
    $arguments = @("evaluate.py") + $eval_args
    $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        Write-Log "Evaluation completed successfully!"
    } else {
        Write-Log "Evaluation failed!"
        exit 1
    }
}
