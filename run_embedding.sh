#!/usr/bin/bash -e


#python calc_embeddings.py --model halft5 --fasta /data/franco/datasets/disprot/monomers.fasta --outdir monomers
#python calc_embeddings.py --model esmb1 --fasta /data/franco/datasets/disprot/monomers.fasta --outdir monomers



#python prepare_plotdata.py --model halft5 --fasta /data/franco/datasets/disprot/monomers.fasta --annot /data/franco/datasets/disprot/monomers_annot.fasta --embeddir monomers/halft5/ --outdir ./monomers_plotdata

python prepare_plotdata.py --model halft5 --fastadir /data/franco/datasets/disprot/fasta/ --annot /data/franco/datasets/disprot/DisProt_release_2022_06_reformat_annot.fasta --embeddir disprot/halft5/ --outdir ./disprot_plotdata

NEIGHBOURS="5 10 30 60 100 200"
DISTS="0.001 0.01 0.1 0.25 0.5 0.9"

for N in $NEIGHBOURS; do
    for D in $DISTS; do
        echo $N $D;
        python calc_umap.py --embedmat ./disprot_plotdata/halft5/embedding_data.txt.npz --neighbours $N --mindist $D --outfile ./disprot_plotdata/halft5/umap_${N}_${D}.txt
    done;
done;
