#!/bin/bash -e

PYTHON=/usr/users/fsimonetti/miniconda3/envs/py39bioembed/bin/python
OUTDIR="/biodata/franco/zsuzsa_lab/loss_T5_human_600aa_1200aa_n1000"

s=0
e=1020

echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_T5.py --fasta jupyter/human_proteome_600_to_1200aa.fasta --start ${s} --end ${e} --outdir ${OUTDIR}