# import bio_embeddings
# from bio_embeddings.embed import ProtTransT5XLU50Embedder,ESM1bEmbedder # ProtTransBertBFDEmbedder #, 
from Bio import SeqIO
from embeddings_config import embedding_data, avail_models
import numpy as np
import os
import argparse
import torch
from tqdm import tqdm

def parse_args():

    parser = argparse.ArgumentParser("Embedder")

    parser.add_argument("--model",
                          type=str,
                          dest="model",
                          help="choose from halft5, prott5 or esmb1 models")

    parser.add_argument("--fastadir",
                          type=str,
                          dest="fastadir",
                          default=None,
                          help="directory of individual fasta files")

    parser.add_argument("--fasta",
                          type=str,
                          default=None,
                          dest="fastafile",
                          help="Multi fasta file")

    parser.add_argument("--upix",
                          type=int,
                          default=1,
                          dest="upix",
                          help="uniprot id position 0|1|2 ..")

    parser.add_argument("--outdir",
                          type=str,
                          dest="outdir",
                          help="Destination directory")

    opts = parser.parse_args()
    return opts

opts = parse_args()
sel_embedding = opts.model #'halft5'

if sel_embedding not in avail_models:
    print("ERROR: Selected model not available")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequences = []
# check for a directory with individual fasta files
# or a multi fasta file
if opts.fastadir is not None:
    fastadir = opts.fastadir # "/data/franco/datasets/disprot/fasta/"
    fastafiles = os.listdir(fastadir)
    for f in fastafiles:
        counter = 0
        for record in SeqIO.parse(os.path.join(fastadir, f), "fasta"):
            sequences.append(record)
            counter += 1
            if counter > 1:
                print("More than one fasta record?", f)
                raise
elif opts.fastafile is not None:
    for record in SeqIO.parse(opts.fastafile, "fasta"):
        sequences.append(record)

#embedder = ProtTransT5XLU50Embedder(model_directory=embedding_data[sel_embedding]['dir'], half_model=True)
if sel_embedding == "halft5":
    embedder = embedding_data[opts.model]["embedder"](model_directory=embedding_data[sel_embedding]['dir'], half_model=True, device=device)
else:
    embedder = embedding_data[opts.model]["embedder"](model_directory=embedding_data[sel_embedding]['dir'], device=device)

output_dir = os.path.join(opts.outdir, sel_embedding)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

upix = opts.upix
for s in tqdm(sequences):
    if "|" in s.name:
        name = s.name.split("|")[upix].strip()
    else:
        name = s.name.strip()
        if name == "":
            print("Name is empty",s.name)
            raise
    print(f"working on {name}")
    aa_sequence = str(s.seq).upper()
    if len(aa_sequence) > 1200:
        print(f"Skipping {name}, len={len(aa_sequence)}")
        continue
    outfile = os.path.join(output_dir, name+".npy")
    if os.path.exists(outfile):
        print(f"File exists: {outfile}")
        continue
    else:
        embedding = embedder.embed(aa_sequence)
        np.save(outfile, embedding)
                                                    
