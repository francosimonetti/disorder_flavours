#!/bin/bash -e

PYTHON=/usr/users/fsimonetti/miniconda3/envs/py311torch2/bin/python
OUTDIR="/biodata/franco/zsuzsa_lab/loss_T5_longseq_autoreg"

s=0
e=11

echo "from ${s} to ${e}"
${PYTHON} calculate_decoder_loss_T5.py --fasta longseq_testset_sequences.fasta --start ${s} --end ${e} --outdir ${OUTDIR} --autoregressive