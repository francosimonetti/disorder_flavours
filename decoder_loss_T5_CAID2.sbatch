#!/bin/bash -e
#SBATCH -p long
#SBATCH -t 4-00:00:00
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -J CAID2_T5_loss_%A_%a
#SBATCH -e logs/CAID2_T5_loss_%A_%a.err
#SBATCH -o logs/CAID2_T5_loss_%A_%a.out



PYTHON=/usr/users/fsimonetti/miniconda3/envs/py39/bin/python
#OUTDIR="/biodata/franco/zsuzsa_lab/loss_T5_CAID2"

s=$(($SLURM_ARRAY_TASK_ID*$BATCH_SIZE)) 
e=$(($SLURM_ARRAY_TASK_ID*$BATCH_SIZE+$BATCH_SIZE))

if [ $e -gt $TOTAL ]; then
    e=$TOTAL
fi
echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_T5.py --fasta ${FASTAFILE} --start ${s} --end ${e} --outdir ${OUTDIR}