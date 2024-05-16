#!/bin/bash -e

export BATCH_SIZE=20
export TOTAL=`grep -c ">" CAID2/caid2_disorder_nox.fasta`
#export TOTAL=100

SHARDS=$(($TOTAL/$BATCH_SIZE))
echo "Splitting job in $SHARDS tasks"


export OUTDIR="loss_T5_CAID2_disorder_nox"
export FASTAFILE="CAID2/caid2_disorder_nox.fasta"
sbatch --array=0-$SHARDS%15 decoder_loss_T5_CAID2.sbatch

# export BATCH_SIZE=10
# export TOTAL=`grep -c ">" CAID2/all_remaining.fasta`
# #export TOTAL=100

# SHARDS=$(($TOTAL/$BATCH_SIZE))
# echo "Splitting job in $SHARDS tasks"


# export OUTDIR="loss_T5_CAID2_all_remaining"
# export FASTAFILE="CAID2/all_remaining.fasta"
# sbatch --array=0-$SHARDS%15 decoder_loss_T5_CAID2.sbatch
# sh decoder_loss_T5_CAID2.sbatch
