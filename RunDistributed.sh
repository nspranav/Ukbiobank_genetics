#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=40g
#SBATCH --gres=gpu:RTX:1
#SBATCH -p qTRDGPU
#SBATCH -J ContrastiveL
#SBATCH -e error%A.err
#SBATCH -o Contrastive_%A.txt
#SBATCH -A trends396s109
#SBATCH -t 0-07:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe


source /data/users3/pnadigapusuresh1/Downloads/anaconda/bin/activate latest3
## python train_lightning.py --job_id ${SLURM_JOBID} --group L1+L2+L4
## python lightning_classification.py --load_from 5335653 --group L1+L2
python Sparse_CCA.py
