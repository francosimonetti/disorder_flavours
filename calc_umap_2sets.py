import umap
import numpy as np
import os
import argparse
import time

def parse_args():

    parser = argparse.ArgumentParser("Calculate UMAP for embedding")

    parser.add_argument("--embedmat1",
                        type=str,
                        dest="embedmatfile1",
                        help="concatenated embedding data for all proteins")
    
    parser.add_argument("--embedmat2",
                        type=str,
                        dest="embedmatfile2",
                        help="concatenated embedding data for all proteins")

    parser.add_argument("--neighbours",
                        type=int,
                        dest="neighbours",
                        help="UMAP neighbours parameter")

    parser.add_argument("--mindist",
                        type=float,
                        dest="mindist",
                        help="UMAP min_dist parameter")

    parser.add_argument("--outfile",
                          type=str,
                          dest="outfile",
                          help="Destination file of UMAP coordinates")

    opts = parser.parse_args()
    return opts

opts = parse_args()

s = time.time()
print("Loading embedding..")
loaded_data = np.load(opts.embedmatfile1)
embeddings1 = loaded_data["concat_emb"]
e = time.time() - s
print(f"Loaded in {e}!")

s = time.time()
print("Loading embedding 2..")
loaded_data = np.load(opts.embedmatfile2)
embeddings2 = loaded_data["concat_emb"]
e = time.time() - s
print(f"Loaded in {e}!")

print(embeddings1.shape, embeddings2.shape)

combined_emb = np.concatenate((embeddings1, embeddings2))

my_umap = umap.UMAP(n_neighbors=opts.neighbours, min_dist=opts.mindist)
umap_embed = my_umap.fit_transform(combined_emb)

np.savetxt(opts.outfile, umap_embed)
