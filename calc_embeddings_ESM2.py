# import bio_embeddings
# from bio_embeddings.embed import ProtTransT5XLU50Embedder,ESM1bEmbedder # ProtTransBertBFDEmbedder #, 
from Bio import SeqIO
import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import esm
import re

def parse_args():

    parser = argparse.ArgumentParser("Embedder")

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\t- Device: {device}")
# Load model and tokenizer
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

if device.type == 'cuda':
    print("Sending to cuda")
    model = model.eval().cuda()  # disables dropout for deterministic results
    #torch.cuda.empty_cache()
else:
    model = model.eval()

batch_converter = alphabet.get_batch_converter()

def embed(target_seq, device=device, model=model, batch_converter=batch_converter):
    input_seq = [" ".join(list(re.sub(r"[UZOB]", "X", target_seq)))]
    batch_labels, batch_strs, batch_tokens = batch_converter([("prot1", input_seq[0])])

    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[36], return_contacts=False)
    token_representations = results["representations"][36].squeeze()[1:-1]
    return token_representations.detach().cpu().numpy()

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

output_dir = os.path.join(opts.outdir)
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
        embedding = embed(aa_sequence)
        np.save(outfile, embedding)
                                                    
