import os
import json
import numpy as np
import torch
import collections
from Bio import SeqIO
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from torch.nn import CrossEntropyLoss
import datetime

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('../models/prottrans_t5_xl_u50', do_lower_case=False, legacy=False)

celoss = CrossEntropyLoss()

T5_vocab = [ i[1] for i in tokenizer.get_vocab().keys() if i.startswith("▁") and not re.search(r"[UZOBX]", i)] # this is not a regular underscore char
T5_vocab_ix = [ j for j,i in enumerate(tokenizer.get_vocab().keys()) if i.startswith("▁") and not re.search(r"[UZOBX]", i)]
print(len(T5_vocab), len(T5_vocab_ix))
print(T5_vocab, T5_vocab_ix)


def get_single_aa_losses(datadict, prot_tokens):
    all_loss_sequences = []
    logits = datadict['aamask_1']['logits']
    for i in range(len(prot_tokens)):
        aaloss_sequence = []
        if datadict['aamask_1']['loss'][i] < 0:
            aaloss_sequence.append(-1)
        else:
            for j in range(len(prot_tokens)):
                loss = celoss(torch.tensor(logits[i][j]).double(), torch.tensor(prot_tokens[j]))   
                aaloss_sequence.append(loss.item())
        all_loss_sequences.append(aaloss_sequence)
    return all_loss_sequences

def get_single_aa_losses_indiv(logits, prot_tokens):
    aaloss_sequence = []
    for j in range(len(prot_tokens)):
        loss = celoss(torch.tensor(logits[j]).double(), torch.tensor(prot_tokens[j]))   
        aaloss_sequence.append(loss.item())
    return aaloss_sequence

def get_position_colors(unip, diso_seq):
    aa_colors = []
    for e in diso_seq:
        if e == "-":
            aa_colors.append('b')
        else:
            aa_colors.append('r')
    return aa_colors

datadict = collections.defaultdict(dict)
for record in SeqIO.parse("../CAID2/caid2_disorder_nox.fasta", "fasta"):
    uniprot_id = record.id
    seq = str(record.seq)
    datadict[uniprot_id]['seq'] = seq

for record in SeqIO.parse("../CAID2/caid2_disorder_nox_annotations.fasta", "fasta"):
    uniprot_id = record.id
    seq = str(record.seq)
    datadict[uniprot_id]['disorder'] = seq

print(f"Loaded {len(datadict.keys())} proteins")

t5dir = "../loss_T5_CAID2_disorder_nox"

for filename in os.listdir(t5dir):
    if filename.endswith(".json"):
        uniprot_id = filename.split(".")[0]
        t5_out   = os.path.join(t5dir, "losses", uniprot_id + "_losses.json")
        t5_matchout = os.path.join(t5dir, "matches", uniprot_id + "_matches.json")
        os.makedirs(os.path.join(t5dir, "aalogits_img"), exist_ok=True)
        logits_img_file = os.path.join(t5dir, "aalogits_img", uniprot_id + "_aalogits_img.npy")
        if not os.path.exists(t5_matchout):
            with open(os.path.join(t5dir, filename)) as f:
                # make a try-exception block to load the json
                try:
                    data = json.load(f)
                except:
                    print(f"ERROR loading {filename}")
                    continue

                t5_dict = data[uniprot_id]

                L = len(datadict[uniprot_id]['seq'])
                this_seq = datadict[uniprot_id]['seq']

                aa_colors = get_position_colors(uniprot_id, datadict[uniprot_id]['disorder'])
                
                print(datetime.datetime.now(), uniprot_id, L)
                ## Tokenize
                input_seq = " ".join(list(re.sub(r"[UZOB]", "X", this_seq)))
                T5_prot_toks = tokenizer(input_seq)['input_ids'][:-1] ## remove end of sequence token
                
                ## Get aa-level losses
                T5_aaloss_sequences   = get_single_aa_losses(t5_dict, T5_prot_toks)
                

                ## get unmasked aa-level losses for the entire sequence
                unmasked_logits_file = os.path.join(t5dir, "logits", uniprot_id + "_logits.pt")
                if os.path.exists(unmasked_logits_file):
                    t5_dict['unmasked_logits'] = torch.load(unmasked_logits_file)
                    T5_unmasked_aaloss_sequence   = get_single_aa_losses_indiv(t5_dict['unmasked_logits'].squeeze(), T5_prot_toks)
                else:
                    raise ValueError(f"Missing unmasked logits for {uniprot_id}")
                
                ## save T5_aaloss_sequences to a file
                os.makedirs(os.path.join(t5dir, "losses"), exist_ok=True)
                with open(os.path.join(t5dir, "losses", uniprot_id + "_losses.json"), "w") as f:
                    json.dump(T5_aaloss_sequences, f)

                ## save T5_unmasked_aaloss_sequence to a file
                with open(os.path.join(t5dir, "losses", uniprot_id + "_unmasked_losses.json"), "w") as f:
                    json.dump(T5_unmasked_aaloss_sequence, f)

                ## save matches to a file
                os.makedirs(os.path.join(t5dir, "matches"), exist_ok=True)
                with open(t5_matchout, "w") as f:
                    json.dump(t5_dict['aamask_1']['match'], f)
                
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

        else:
            print(f"Skipping {uniprot_id}")
