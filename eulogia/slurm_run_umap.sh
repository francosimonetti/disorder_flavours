#!/bin/bash
#SBATCH -J umap_disprot
#SBATCH --partition=multi
#SBATCH --time=2-00:00
#SBATCH -n 256
NEIGHBOURS="200"
DISTS="0.001 0.01 0.1 0.25 0.5 0.9"

for N in $NEIGHBOURS; do
    for D in $DISTS; do
        echo $N $D;
        python calc_umap.py --embedmat ./disprot_plotdata/halft5/embedding_data.txt.npz --neighbours $N --mindist $D --outfile ./disprot_plotdata/halft5/umap_${N}_${D}.txt
    done;
done;

