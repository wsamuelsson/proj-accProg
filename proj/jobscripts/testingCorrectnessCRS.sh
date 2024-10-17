#!/bin/bash -l
#SBATCH -A uppmax2024-2-16 # project name 
#SBATCH -M snowy # name of system 
#SBATCH -p node # request a full node 
#SBATCH -N 1 # request 1 node 
#SBATCH -t 0:15:00 # job takes at most 1 hour 
#SBATCH --gres=gpu:1 --gpus-per-node=1 # use the GPU nodes
#SBATCH -J testingCorrectnessCRS # name of the job 
#SBATCH -D ./ # stay in current working directory 
#SBATCH -o ../results/testingCorrectnessCRS.txt

nvidia-smi    #GPU info

./../test/testingCRS
