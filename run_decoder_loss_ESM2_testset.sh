#!/bin/bash

export BATCH_SIZE=1
export TOTAL=10

SHARDS=$(($TOTAL/$BATCH_SIZE))
echo "Splitting job in $SHARDS tasks"


sbatch --array=0-$SHARDS%15 decoder_loss_ESM2_testset.sbatch