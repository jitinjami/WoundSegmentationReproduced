#!/bin/bash -l
#SBATCH --clusters=tinygpu
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=linknet

module load cuda/11.8.0
module load python/3.8-anaconda
conda activate seg
cd /home/hpc/iwso/iwso114h/WoundSegmentationReproduced/
python3 main.py --no-mnv2 --train --test
