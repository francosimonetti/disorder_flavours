#!/bin/bash -e

PYTHON=/usr/users/fsimonetti/miniconda3/envs/py39bioembed/bin/python
OUTDIR="/biodata/franco/zsuzsa_lab/loss_T5_disprot2_cuda"

s=0
e=2500

echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_T5.py --fasta disprot_OK_fullset_12_2023.fasta --start ${s} --end ${e} --outdir ${OUTDIR}