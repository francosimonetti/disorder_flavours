import umap
import numpy as np
import os
import argparse
import time

def parse_args():

    parser = argparse.ArgumentParser("Calculate UMAP for embedding")

    parser.add_argument("--embedmat",
                        type=str,
                        dest="embedmatfile",
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
loaded_data = np.load(opts.embedmatfile)
embeddings = loaded_data["concat_emb"]
e = time.time() - s
print(f"Loaded in {e}!")

my_umap = umap.UMAP(n_neighbors=opts.neighbours, min_dist=opts.mindist)
umap_embed = my_umap.fit_transform(embeddings)

outdir = os.path.dirname(opts.outfile)
if not os.path.exists(outdir):
    os.makedirs(outdir)
np.savetxt(opts.outfile, umap_embed)
