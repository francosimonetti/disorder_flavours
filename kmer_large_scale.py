from Bio import SeqIO
from embeddings_config import avail_models
import numpy as np
import os
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import mpl_stylesheet
mpl_stylesheet.banskt_presentation(fontfamily = 'mono', fontsize = 20, colors = 'banskt', dpi = 300)

def get_kmers(seq, K=30):
    start = 0
    end   = len(seq)
    kmers = list()
    ranges= list()
    for i in range(start, end):
        if i > len(seq)-K or ((i+K)>len(seq)):
            break
        else:
            kmers.append(seq[i:i+K])
            ranges.append(f"{i}_{i+K}")
    return kmers, ranges

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

#target_uniprots = ["P37840", "P04637", "P02686", "P07305", "O00488", "Q9NYB9", "P06401", "Q16186", "S6B291", "P23441"]

dataset_name="disprot"
sel_embedding = 'halft5'
upix = 1
Ks = [30]
fastafile=None
fastadir="/data/franco/datasets/disprot/fasta/"
annotdir=None
annotfile="/data/franco/datasets/disprot/DisProt_release_2022_06_reformat_annot.fasta"

if sel_embedding not in avail_models:
    print("ERROR: Selected model not available")
    raise

sequences = get_sequences(fastadir=fastadir, fastafile=fastafile)
annots    = get_sequences(fastadir=annotdir, fastafile=annotfile)

# find annotation for each sequence
# read annotation data first
annot_dict = dict()
for record in annots:
    if "|" in record.name:
        name = record.name.split("|")[upix].strip()
    else:
        name = record.name.strip()
        if name == "":
            print("Name is empty",record.name)
            raise
    # if name != "A0A6L8PPD0":
    #     continue
    annot_dict[name] = str(record.seq)

embeddings = []
selected_sequences = []
disseqs = list()
selected_uniprots = list()

for s in sequences:
    if "|" in s.name:
        name = s.name.split("|")[upix].strip()
    else:
        name = s.name.strip()
        if name == "":
            print("Name is empty",s.name)
            raise
    #if name in target_uniprots:
    print(f"Loading {name} data")
    aa_sequence = str(s.seq).upper()
    if len(aa_sequence) > 1200:
        print(f"Skipping {name}, len={len(aa_sequence)}")
        continue
    if len(s.seq) == len(annot_dict[name]):
            disseqs.append(annot_dict[name])
            selected_sequences.append(aa_sequence)
            selected_uniprots.append(name)
    else:
        print("Sequence and annotation length do not match")
        print(len(s.seq), len(annot_dict[name]), name)

print("Generating dataset kmer file..")
for K in Ks:
    print(K)
    dataset_kmers_file = f"kmer_analysis_data/{dataset_name}_kmers{K}.fasta"
    with open(dataset_kmers_file, 'w') as outstream:
        for i,s in enumerate(selected_sequences):
            uniprot_id = selected_uniprots[i]
            # kmers_file = f"kmer_analysis_data/{uniprot_id}_kmers{K}.fasta"
            kmers, ranges = get_kmers(s, K=K)
            for kmer, pos in zip(kmers, ranges):
                #print(pos, kmer)
                #print(f">{uniprots[0]}_{pos}")
                outstream.write(f">{uniprot_id}_{pos}\n")
                outstream.write(f"{kmer}\n")

print("Begin sequence embedding process..")   
for K in Ks:
    #for u in selected_uniprots:
    dataset_kmers_file = f"kmer_analysis_data/{dataset_name}_kmers{K}.fasta"
    cmd = f"python calc_embeddings.py --model halft5 --fasta {dataset_kmers_file} --upix 0 --outdir {dataset_name}_kmers{K}"
    print(cmd)
    os.system(cmd)


# # must redo this part
# kdict = dict()
# for K in Ks:
#     print(K)
#     sequences = []
#     annotations = []
#     embeddings = []
#     kmers_per_prot = []
#     dataset_kmers_file = f"kmer_analysis_data/{dataset_name}_kmers{K}.fasta"
#     for i,s in enumerate(selected_sequences):
#         uniprot_id = selected_uniprots[i]
#         kmers_file = f"kmer_analysis_data/{uniprot_id}_kmers{K}.fasta"
#         if os.path.exists(kmers_file):
#             records = get_sequences(fastadir=None, fastafile=kmers_file)
#             this_sequences = []
#             this_embeddings = []
#             for r in records:
#                 kmer_id = r.id
#                 embedfile = f"kmer_analysis_data/{uniprot_id}_kmers{K}/{sel_embedding}/{kmer_id}.gz"
#                 if os.path.exists(embedfile):
#                     this_sequences.append(str(r.seq))
#                     this_embeddings.append(np.loadtxt(embedfile))
#                 else:
#                     print(f"File does not exist:{embedfile}")
#             annotmers, ranges = get_kmers(disseqs[i], K=K)
#             if len(annotmers) == len(this_embeddings) and len(annotmers) == len(this_sequences):
#                 annotations = annotations + annotmers
#                 sequences = sequences + this_sequences
#                 embeddings = embeddings + this_embeddings
#                 kmers_per_prot = kmers_per_prot + [len(this_sequences)]
#             else:
#                 print(f"Some length does not match: {uniprot_id},{i}")
#                 print(len(annotmers), len(this_embeddings), len(this_sequences))
#                 raise
#     kdict[K]=dict()
#     kdict[K]['seqs']  =sequences
#     kdict[K]['annot'] =annotations
#     kdict[K]['embded']=embeddings
#     kdict[K]['n_kmers']=kmers_per_prot


# for K in Ks:
#     kmer_whole_embeddings = []
#     for e in kdict[K]['embded']:
#         kmer_whole_embeddings.append(np.mean(e, axis=0))

#     kmer_disorder_contents = []
#     for da in kdict[K]['annot']:
#         contents = [ x != "-" for x in da]
#         DC = np.sum(contents) / len(contents)
#         kmer_disorder_contents.append(DC)

#     print("Calculaiting UMAP...")
#     my_umap = umap.UMAP(n_neighbors=200, min_dist=.25)
#     umap_kmer_embedding = my_umap.fit_transform(np.array(kmer_whole_embeddings))

#     print("Plotting..")
#     fig = plt.figure(figsize=(8,6))
#     ax1  = fig.add_subplot(111)

#     x = umap_kmer_embedding[:,0]
#     y = umap_kmer_embedding[:,1]

#     cmap = plt.get_cmap("coolwarm")
#     sc = ax1.scatter(x, y, s=2, c=kmer_disorder_contents, cmap=cmap)
#     ax1.set_xlabel("UMAP1")
#     ax1.set_ylabel("UMAP2")
#     ax1.set_title(f"True kmers embeddings, K={K}")
#     plt.colorbar(sc, ax=ax1)
#     plt.savefig(f"plots/embedded_kmers_k{K}.png", bbox_inches='tight')
#     # plt.show()

#     prot_colors = []
#     for i,l in enumerate(kdict[K]['n_kmers']):
#         prot_colors = prot_colors + list(np.repeat(i/len(kdict[K]['n_kmers']), l))

    
#     fig = plt.figure(figsize=(8,6))
#     ax1  = fig.add_subplot(111)

#     x = umap_kmer_embedding[:,0]
#     y = umap_kmer_embedding[:,1]

#     cmap = plt.get_cmap("tab10")
#     sc = ax1.scatter(x, y, s=2, c=prot_colors, cmap=cmap)
#     ax1.set_xlabel("UMAP1")
#     ax1.set_ylabel("UMAP2")
#     ax1.set_title(f"True kmers embeddings, K={K}")
#     ax1.legend()
#     plt.savefig(f"plots/embedded_kmers_k{K}_byprot.png", bbox_inches='tight')