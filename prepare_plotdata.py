from Bio import SeqIO
from embeddings_config import avail_models
import numpy as np
import os
import argparse

def parse_args():

    parser = argparse.ArgumentParser("Prepare embeddings data for doing UMAP")

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

    parser.add_argument("--annotdir",
                          type=str,
                          dest="annotdir",
                          default=None,
                          help="directory of individual fasta annotation files")

    parser.add_argument("--annot",
                          type=str,
                          default=None,
                          dest="annotfile",
                          help="Multi fasta file with annotations")

    parser.add_argument("--embeddir",
                        type=str,
                        dest="embeddir",
                        help="directory with embedding data (outdir from calc_embeddings.py)")

    parser.add_argument("--outdir",
                          type=str,
                          dest="outdir",
                          help="Destination directory of plots")

    opts = parser.parse_args()
    return opts

def get_sequences(fastadir=None, fastafile=None):
    sequences = []
    if fastadir is None and fastafile is None:
        print("No fasta dir or file")
        raise
    if fastadir is not None and fastafile is not None:
        print("Choose one, fasta dir or multi fasta filr")
        raise
    # check for a directory with individual fasta files
    # or a multi fasta file
    if fastadir is not None:
        fastafiles = os.listdir(fastadir)
        for f in fastafiles:
            counter = 0
            for record in SeqIO.parse(os.path.join(fastadir, f), "fasta"):
                sequences.append(record)
                counter += 1
                if counter > 1:
                    print("More than one fasta record?", f)
                    raise
    elif fastafile is not None:
        for record in SeqIO.parse(fastafile, "fasta"):
            sequences.append(record)
    return sequences

opts = parse_args()
sel_embedding = opts.model #'halft5'

if sel_embedding not in avail_models:
    print("ERROR: Selected model not available")
    raise


sequences = get_sequences(fastadir=opts.fastadir, fastafile=opts.fastafile)
annots    = get_sequences(fastadir=opts.annotdir, fastafile=opts.annotfile)

# find annotation for each sequence
# read annotation data first
annot_dict = dict()
for record in annots:
    if "|" in record.name:
        name = record.name.split("|")[1].strip()
    else:
        name = record.name.split()[0].strip()
        if name == "":
            print("Name is empty",record.name)
    # if name != "A0A6L8PPD0":
    #     continue
    annot_dict[name] = str(record.seq)

embeddings = []
selected_sequences = []
disseqs = list()

for s in sequences:
    if "|" in s.name:
        name = s.name.split("|")[1].strip()
    else:
        name = s.name.split()[0].strip()
        if name == "":
            print("Name is empty",s.name)
            raise
    # if name != "A0A6L8PPD0":
    #     continue
    print(f"working on {name}")
    aa_sequence = str(s.seq).upper()
    if len(aa_sequence) > 1200:
        print(f"Skipping {name}, len={len(aa_sequence)}")
        continue
    embed_file = os.path.join(opts.embeddir, name+".gz")
    if os.path.exists(embed_file):
        if len(s.seq) == len(annot_dict[name]):
            # read embedding
            e = np.loadtxt(embed_file)
            # add annot, seq and embedding
            if len(s.seq) == e.shape[0]:
                disseqs.append(annot_dict[name])
                embeddings.append(e)
                selected_sequences.append(aa_sequence)
            else:
                print("Embedding length and seq do not match")
                print(e.shape, len(s.seq), name)
        else:
            print("Sequence and annotation length do not match")
            print(len(s.seq), len(annot_dict[name]), name)
    else:
        ## skip this file
        print(f"skipped {embed_file} does not exist")
        continue
        # associate with the record sequence
        # associate with the disorder/order annotation for coloring

# Concatenate embedding data
concat_emb = np.vstack(embeddings)

# Concatenate sequence aminoacids and annotations
ss = [s for s in selected_sequences]
concat_seq = [aa for seq in ss for aa in seq]
concat_dis = [dd for disseq in disseqs for dd in disseq]
print(len(concat_dis), len(concat_seq))

output_dir = os.path.join(opts.outdir, sel_embedding)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save data for sequence, embeddings, and annotations
embed_out = os.path.join(opts.outdir, sel_embedding, "embedding_data.txt")
seq_out   = os.path.join(opts.outdir, sel_embedding, "seq_data.txt")
annot_out = os.path.join(opts.outdir, sel_embedding, "annot_data.txt")

def list_saver(mylist, outfile):
    with open(outfile, 'w') as outfmt:
        for e in mylist:
            outfmt.write(e)
    return None

print("Saving embedding data")
#np.savetxt(embed_out, np.array(concat_emb))
np.savez_compressed(embed_out, concat_emb=np.array(concat_emb))
print("Saving seq data")
list_saver(concat_seq, seq_out)
print("Saving annot data")
list_saver(concat_dis, annot_out)