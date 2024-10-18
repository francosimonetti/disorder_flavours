#!/bin/bash -e

PYTHON=/usr/users/fsimonetti/miniconda3/envs/py39bioembed/bin/python
OUTDIR="/biodata/franco/zsuzsa_lab/loss_T5_human_max600aa_n2000"

s=0
e=2020

echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_T5.py --fasta jupyter/human_proteome_max600aa.fasta --start ${s} --end ${e} --outdir ${OUTDIR}