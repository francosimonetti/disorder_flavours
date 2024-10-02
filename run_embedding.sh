#!/usr/bin/bash -e

#### 1. Calculate embeddings
## obtain embeddings for a fasta dir or multi fasta file

#python calc_embeddings.py --model halft5 --fastadir /data/franco/datasets/disprot/fasta --outdir disprot
#python calc_embeddings.py --model halft5 --fasta /data/franco/datasets/disprot/monomers.fasta --outdir /data/franco/datasets/prot_embedding_weights/monomers
#python calc_embeddings.py --model esmb1 --fasta /data/franco/datasets/disprot/monomers.fasta --outdir monomers

#python calc_embeddings.py --model halft5 --upix 0 --fasta /data/franco/datasets/disprot/disprot_regions_seq.fasta --outdir /data/franco/datasets/prot_embedding_weights/disprot_regions

python calc_embeddings.py --model halft5 --upix 0 --fasta /biodata/franco/datasets/disprot/all_disprot_seq_and_regions_concat_OK_2023_12.fasta --outdir ./disprot_2023_12_embeddings

python calc_embeddings_ESM2.py --upix 0 --fasta /biodata/franco/datasets/disprot/all_disprot_seq_and_regions_concat_OK_2023_12.fasta --outdir ./disprot_2023_12_embeddings_ESM2

#### 2. Prepare data
## Prepares seq_data, annot_data and embedding data by concatenating sequences, annotations and embedding matrices
## Makes sure that the fasta matches dimensions of the annotations and embeddings

#python prepare_plotdata.py --model halft5 --fasta /data/franco/datasets/disprot/monomers.fasta --annot /data/franco/datasets/disprot/monomers_annot.fasta --embeddir /data/franco/datasets/prot_embedding_weights/monomers/halft5/ --outdir ./monomers_plotdata
#python prepare_plotdata.py --model halft5 --fastadir /data/franco/datasets/disprot/fasta/ --annot /data/franco/datasets/disprot/DisProt_release_2022_06_reformat_annot.fasta --embeddir /data/franco/datasets/prot_embedding_weights/disprot/halft5/ --outdir ./disprot_plotdata
#python prepare_plotdata.py --model halft5 --upix 0 --fasta /data/franco/datasets/disprot/disprot_regions_seq.fasta --annot /data/franco/datasets/disprot/disprot_regions_annot.fasta --embeddir /data/franco/datasets/prot_embedding_weights/disprot_regions/halft5/ --outdir /data/franco/disorder_flavours/disprot_regions_plotdata

#Compile data for match-mismatch analysis
python prepare_plotdata.py --model halft5 --fasta disprot_OK_fullset.fasta --annot disprot_OK_fullset_annotations.fasta --embeddir /biodata/franco/datasets/prot_embedding_weights/disprot/halft5 --outdir disprot_plotdata_OK_fullset

#### 3. Calculate UMAP
### Don't run just now
## This was run on the cluster, takes too long

# NEIGHBOURS="5 10 30 60 100 200"
# DISTS="0.001 0.01 0.1 0.25 0.5 0.9"

# for N in $NEIGHBOURS; do
#     for D in $DISTS; do
#         echo $N $D;
#         python calc_umap.py --embedmat ./disprot_plotdata/halft5/embedding_data.txt.npz --neighbours $N --mindist $D --outfile ./disprot_plotdata/halft5/umap_${N}_${D}.txt
#     done;
# done;

########################

#### 4. Make UMAP plots
## load UMAP coordinates from previous step and plot scatter plots with seq and annot data

