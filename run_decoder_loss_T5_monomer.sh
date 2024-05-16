#!/bin/bash -e

export BATCH_SIZE=10
#export TOTAL=`wc -l monomer_OK_fullset.uniprot_ids.txt | cut -d " " -f 1`
export TOTAL=100

SHARDS=$(($TOTAL/$BATCH_SIZE))
echo "Splitting job in $SHARDS tasks"


sbatch --array=0-$SHARDS%23 decoder_loss_T5_monomer.sbatch
