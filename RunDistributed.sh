#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=10g
#SBATCH --gres=gpu:V100:1
#SBATCH -p qTRDGPUH
#SBATCH -J ConvLr00001
#SBATCH -e error%A.err
#SBATCH -o L1regular_%A.txt
#SBATCH -A trends396s109
#SBATCH -t 2-05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe


source /data/users2/pnadigapusuresh1/software/bin/activate workinglatest
python train.py ${SLURM_JOBID}
