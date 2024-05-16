#!/bin/bash -e
#SBATCH -p long
#SBATCH -t 4-00:00:00
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -J loss_ESM2_testset_%A_%a
#SBATCH -e logs/loss_ESM2_testset_%A_%a.err
#SBATCH -o logs/loss_ESM2_testset_%A_%a.out



PYTHON=/usr/users/fsimonetti/miniconda3/envs/py39/bin/python
OUTDIR="/biodata/franco/zsuzsa_lab/loss_testset_ESM2"

s=$(($SLURM_ARRAY_TASK_ID*$BATCH_SIZE)) 
e=$(($SLURM_ARRAY_TASK_ID*$BATCH_SIZE+$BATCH_SIZE))

if [ $e -gt $TOTAL ]; then
    e=$TOTAL
fi
echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_ESM2.py --fasta testset_sequences.fasta --start ${s} --end ${e} --outdir ${OUTDIR}