#!/bin/bash -e

PYTHON=/usr/users/fsimonetti/miniconda3/envs/py39bioembed/bin/python
OUTDIR="/biodata/franco/zsuzsa_lab/loss_T5_monomer2"

s=0
e=4000

echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_T5.py --fasta monomer_OK_fullset.fasta --start ${s} --end ${e} --outdir ${OUTDIR}