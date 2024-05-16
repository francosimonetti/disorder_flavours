#!/bin/bash -e

export BATCH_SIZE=10
export TOTAL=100

SHARDS=$(($TOTAL/$BATCH_SIZE))
echo "Splitting job in $SHARDS tasks"


sbatch --array=0-$SHARDS%15 decoder_loss_ESM2_monomer.sbatch