#!/bin/bash -e


PYTHON=/usr/users/fsimonetti/miniconda3/envs/py39/bin/python
OUTDIR="/biodata/franco/zsuzsa_lab/loss_ESM2_disprot2"

s=0
e=3000

echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_ESM2.py --fasta /biodata/franco/datasets/disprot/disprot_OK_fullset_2023_12.fasta --start ${s} --end ${e} --outdir ${OUTDIR}
