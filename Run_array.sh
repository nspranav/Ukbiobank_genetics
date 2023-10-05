#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=10g
#SBATCH --gres=gpu:V100:1
#SBATCH -p qTRDGPUH
#SBATCH -J param_tuning
#SBATCH -e error%A-%a.err
#SBATCH -o BioInformed_%A-%a.txt
#SBATCH -A trends396s109
#SBATCH -t 5-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe

source /data/users2/pnadigapusuresh1/software/bin/activate workinglatest
python cross_val.py ${SLURM_JOBID} ${SLURM_ARRAY_TASK_ID}
