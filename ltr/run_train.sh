#!/bin/bash
#SBATCH -J transformer
#SBATCH --output=/home/yans/pytracking/ltr/logs/log-output.txt
#SBATCH --error=/home/yans/pytracking/ltr/logs/log-error.txt
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu --gres=gpu:1

module load CUDA/10.0
module load fgci-common
module load ninja/1.9.0

conda activate pytracking
cd /home/yans/pytracking/ltr/
python run_training.py transformer transformer50

conda deactivate
