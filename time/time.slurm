#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH -t 23:59:59
#SBATCH -J voids

srun --multi-prog time.conf
