from bio_embeddings.embed import ProtTransT5XLU50Embedder,ESM1bEmbedder # ProtTransBertBFDEmbedder #, 

avail_models = ["halft5", "prott5", "esmb1"]
halft5_dir = "/data/franco/datasets/prot_embedding_weights/half_prottrans_t5_xl_u50/"
prott5_dir = "/data/franco/datasets/prot_embedding_weights/prottrans_t5_xl_u50/"
esmb1_dir  = "/data/franco/datasets/prot_embedding_weights/"

embedding_data = {}
embedding_data['halft5'] = { 'dir': halft5_dir, "embedder": ProtTransT5XLU50Embedder }
embedding_data['prott5'] = { 'dir': prott5_dir, "embedder": ProtTransT5XLU50Embedder }
embedding_data['esmb1']  = { 'dir': esmb1_dir,  "embedder": ESM1bEmbedder}