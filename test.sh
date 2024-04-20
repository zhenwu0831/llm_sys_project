#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/home/zhenwu/11868/logs/bashout.log

python project/dataset.py