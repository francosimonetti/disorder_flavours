#!/bin/bash -e

export BATCH_SIZE=15
export TOTAL=`wc -l disprot_OK_shortseq.uniprot_ids.txt | cut -d " " -f 1`
#export TOTAL=100

SHARDS=$(($TOTAL/$BATCH_SIZE))
echo "Splitting job in $SHARDS tasks"


sbatch --array=0-$SHARDS%23 decoder_loss_T5_disprot_short_cpu_autoreg.sbatch