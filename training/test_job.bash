#!/bin/bash
#SBATCH --job-name=test_withWANDB
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2

#module load SciPy-bundle/2020.11-fosscuda-2020b
#module load h5py/2.10.0-fosscuda-2020a-Python-3.8.2
#module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
#module load matplotlib/3.3.3-fosscuda-2020b
module load scikit-learn/1.0.1-foss-2021b 
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1 
module load matplotlib/3.4.3-foss-2021b 
module load wandb/0.12.16-GCCcore-11.2.0
module load tensorflow_addons/0.15.0-foss-2021b-CUDA-11.4.1


python light_test.py

