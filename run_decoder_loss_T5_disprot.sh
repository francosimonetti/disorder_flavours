#!/bin/bash -e

export BATCH_SIZE=20
export TOTAL=`wc -l disprot_OK_fullset.uniprot_ids.txt | cut -d " " -f 1`
#export TOTAL=100

SHARDS=$(($TOTAL/$BATCH_SIZE))
echo "Splitting job in $SHARDS tasks"


sbatch --array=0-$SHARDS%15 decoder_loss_T5_disprot.sbatch

# for (( i=0; i<=${SHARDS}; i++ )); do
#     s=$(($i*$BATCH_SIZE))
#     e=$(($i*$BATCH_SIZE+$BATCH_SIZE))
    
    
#     if [ $e -gt $TOTAL ]; then
#         e=$TOTAL
#     fi
#     sbatch -p long -t "4-00:00:00" -c 16 -N 1 -J "loss_${s}_${e}" --exclusive -e "logs/%x.err" -o "logs/%x.log" \
#         --wrap="${PYTHON} calculate_decoder_loss.py --model models/prottrans_t5_xl_u50/ --fasta disprot_OK_fullset.fasta --start ${s} --end ${e} --outdir ${OUTDIR}";
#     sleep 1;
# done;