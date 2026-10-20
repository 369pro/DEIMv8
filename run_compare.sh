#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=/home/scnu2024024530/WZW/DEIMv8/compare_imgs/slurm_%j.out
#SBATCH --error=/home/scnu2024024530/WZW/DEIMv8/compare_imgs/slurm_%j.err

cd /home/scnu2024024530/WZW/DEIMv8
export PATH="/home/scnu2024024530/.conda/envs/py310/bin:/home/apps/miniconda3/bin:$PATH"

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

python batch_compare.py
