#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=10g
#SBATCH -p qTRD
#SBATCH -J param_tuning
#SBATCH -e error%A-%a.err
#SBATCH -o emebedding_%A-%a.txt
#SBATCH -A trends396s109
#SBATCH -t 5-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe

source /data/users3/pnadigapusuresh1/software/bin/activate workinglatest
python cross_val.py --job_id ${SLURM_JOBID} ${SLURM_ARRAY_TASK_ID}
