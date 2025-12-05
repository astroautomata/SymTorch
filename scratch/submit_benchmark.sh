#!/bin/bash
#SBATCH --job-name=symtorch_benchmark
#SBATCH --output=/cephfs/store/gr-mc2473/as3591/code/SymTorch/scratch/slurm/benchmark_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere

# Print GPU info
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi
echo ""

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the benchmark
python /cephfs/store/gr-mc2473/as3591/code/SymTorch/scratch/benchmark_symbolicmodel.py "$@"

echo "\nJob completed!"
