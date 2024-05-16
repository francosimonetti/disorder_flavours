import os
import json
import numpy as np
import torch
import collections
from Bio import SeqIO
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from tqdm import tqdm

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('../models/prottrans_t5_xl_u50', do_lower_case=False, legacy=False)

T5_vocab = [ i[1] for i in tokenizer.get_vocab().keys() if i.startswith("▁") and not re.search(r"[UZOBX]", i)] # this is not a regular underscore char
T5_vocab_ix = [ j for j,i in enumerate(tokenizer.get_vocab().keys()) if i.startswith("▁") and not re.search(r"[UZOBX]", i)]
print(len(T5_vocab), len(T5_vocab_ix))
print(T5_vocab, T5_vocab_ix)


datadict = collections.defaultdict(dict)
for record in SeqIO.parse("../disprot_OK_fullset_12_2023.fasta", "fasta"):
    uniprot_id = record.id
    seq = str(record.seq)
    datadict[uniprot_id]['seq'] = seq

for record in SeqIO.parse("../disprot_OK_fullset_annotations_12_2023.fasta", "fasta"):
    uniprot_id = record.id
    seq = str(record.seq)
    datadict[uniprot_id]['disorder'] = seq

print(f"Loaded {len(datadict.keys())} proteins")

t5dir_disprot = "../loss_T5_disprot"

for filename in tqdm(os.listdir(t5dir_disprot)):
    if filename.endswith(".json"):
        uniprot_id = filename.split(".")[0]
        if uniprot_id in datadict:
            os.makedirs(os.path.join(t5dir_disprot, "aalogits_img"), exist_ok=True)
            logits_img_file = os.path.join(t5dir_disprot, "aalogits_img", uniprot_id + "_aalogits_img.npy")    
            if os.path.exists(logits_img_file):
                continue

            if not os.path.join(t5dir_disprot, filename):
                print(f"Missing {uniprot_id}")
                continue
            
            with open(os.path.join(t5dir_disprot, filename)) as f:
                print(uniprot_id, end=" ")

                data = json.load(f)
                t5_dict = data[uniprot_id]

                L = len(datadict[uniprot_id]['seq'])
                print(L)
                this_seq = datadict[uniprot_id]['seq']
                
                ## Tokenize
                input_seq = " ".join(list(re.sub(r"[UZOB]", "X", this_seq)))
                T5_prot_toks = tokenizer(input_seq)['input_ids'][:-1] ## remove end of sequence token              

                ## get unmasked aa-level losses for the entire sequence
                unmasked_logits_file = os.path.join(t5dir_disprot, "logits", uniprot_id + "_logits.pt")
                if os.path.exists(unmasked_logits_file):
                    t5_dict['unmasked_logits'] = torch.load(unmasked_logits_file)
                else:
                    raise ValueError(f"Missing unmasked logits for {uniprot_id}")
                
                print(f"Saving logits img file {uniprot_id}")
                # gather all aa-level losses into a single matrix
                all_aalogits = []
                for i in range(len(T5_prot_toks)):
                    all_aalogits.append(np.array(t5_dict['aamask_1']['logits'][i][i]))
                all_aalogits = np.array(all_aalogits)[:,T5_vocab_ix]
                
                # stack both logit matrices like an image
                aalogits_img = np.stack([all_aalogits, t5_dict['unmasked_logits'].squeeze()[:,T5_vocab_ix].numpy()])
                aalogits_img = aalogits_img.transpose(1,2,0)

                # save all_aalogits to file  
                np.save(logits_img_file, aalogits_img)