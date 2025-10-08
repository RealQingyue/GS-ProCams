# =============================================================================
# GS-ProCams Pipeline for Synthetic Datasets - PowerShell Version
# =============================================================================
# This script runs the complete pipeline: training, rendering, and evaluation
# For easy reproduction of the results
# =============================================================================

# Configuration
$input_dir = if ($env:INPUT_DIR) { $env:INPUT_DIR } else { "data/nepmap-dataset" }
$output_dir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "output/nepmap-dataset" }

$setup_names = @("castle", "planck", "zoo", "pear")
$model_types = @("wo_psf")

$gpu_id = if ($env:GPU_ID) { $env:GPU_ID } else { "0" }

# Function to monitor GPU memory usage
function Monitor-GPUMemory {
    param(
        [string]$LogDir,
        [string]$GPUId
    )
    
    # Create directory if it doesn't exist
    if (!(Test-Path -Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    
    $log_file = Join-Path $LogDir "gpu_memory.log"
    $start_time = Get-Date
    $counter = 0
    
    Add-Content -Path $log_file -Value "Starting GPU memory monitoring for GPU $GPUId"
    Add-Content -Path $log_file -Value "Time, Runtime, GPU Memory(MB)"
    
    while ($true) {
        $current_time = Get-Date
        $elapsed = $current_time - $start_time
        $elapsed_minutes = [math]::Floor($elapsed.TotalMinutes)
        $remaining_seconds = $elapsed.Seconds
        
        try {
            $memory_usage = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$GPUId
        }
        catch {
            Write-Error "Failed to get GPU memory usage"
            break
        }
        
        $counter++
        $time_str = $current_time.ToString("HH:mm:ss")
        Add-Content -Path $log_file -Value "$time_str, ${elapsed_minutes}min${remaining_seconds}s, ${memory_usage}MB"
        
        Start-Sleep -Seconds 10
    }
}

# Training loop
foreach ($setup_name in $setup_names) {
    foreach ($model_type in $model_types) {
        # Set model directory and extra arguments based on model type
        switch ($model_type) {
            "wo_psf" {
                $model_dir = Join-Path $output_dir "wo_psf\setups\$setup_name"
                $extra_args = @("--wo_psf", "--white_background")
            }
            default {
                Write-Error "Error: Unknown model type: $model_type"
                exit 1
            }
        }

        # Start GPU memory monitoring in the background
        if ([string]::IsNullOrEmpty($model_dir)) {
            Write-Error "Error: model_dir is empty"
            exit 1
        }

        # Start GPU memory monitoring as a background job
        $monitor_job = Start-Job -ScriptBlock ${function:Monitor-GPUMemory} -ArgumentList $model_dir, $gpu_id

        try {
            # Run the training script
            Write-Host "Run train.py $($extra_args -join ' ') for: $setup_name"
            $input_path = Join-Path $input_dir $setup_name
            
            $arguments = @("train.py", "-s", $input_path, "-m", $model_dir, "--evaluate") + $extra_args + @("--gpu_id", $gpu_id)
            $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
            
            if ($process.ExitCode -ne 0) {
                Write-Error "Training failed for $setup_name"
            }
        }
        finally {
            # Stop GPU memory monitoring
            Stop-Job -Job $monitor_job
            Remove-Job -Job $monitor_job
        }
    }
}
