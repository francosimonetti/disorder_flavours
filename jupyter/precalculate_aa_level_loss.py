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
import esm
import datetime

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('../models/prottrans_t5_xl_u50', do_lower_case=False, legacy=False)

# Load model and tokenizer (Avoid loading entire model)
#model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

celoss = CrossEntropyLoss()
batch_converter = alphabet.get_batch_converter()

T5_vocab = [ i[1] for i in tokenizer.get_vocab().keys() if i.startswith("▁") and not re.search(r"[UZOBX]", i)] # this is not a regular underscore char
T5_vocab_ix = [ j for j,i in enumerate(tokenizer.get_vocab().keys()) if i.startswith("▁") and not re.search(r"[UZOBX]", i)]
print(len(T5_vocab), len(T5_vocab_ix))
print(T5_vocab, T5_vocab_ix)

ESM2_vocab = [i for i in alphabet.all_toks if len(i) == 1 and not re.search(r"[UZOBX\.-]", i)]
ESM2_vocab_ix = [j for j,i in enumerate(alphabet.all_toks) if len(i) == 1 and not re.search(r"[UZOBX\.-]", i)]
print(len(ESM2_vocab))
print(ESM2_vocab,  ESM2_vocab_ix)


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
for record in SeqIO.parse("../disprot_OK_fullset.fasta", "fasta"):
    uniprot_id = record.id
    seq = str(record.seq)
    datadict[uniprot_id]['seq'] = seq

for record in SeqIO.parse("../disprot_OK_fullset_annotations.fasta", "fasta"):
    uniprot_id = record.id
    seq = str(record.seq)
    datadict[uniprot_id]['disorder'] = seq

print(f"Loaded {len(datadict.keys())} proteins")

t5dir_disprot = "../loss_T5_disprot"
esm2dir = "../loss_ESM2_disprot"

for filename in os.listdir(t5dir_disprot):
    if filename.endswith(".json"):
        uniprot_id = filename.split(".")[0]
        t5_out   = os.path.join(t5dir_disprot, "losses", uniprot_id + "_losses.json")
        esm2_out = os.path.join(esm2dir, "losses", uniprot_id + "_losses.json")
        t5_matchout = os.path.join(t5dir_disprot, "matches", uniprot_id + "_matches.json")
        esm2_matchout = os.path.join(esm2dir, "matches", uniprot_id + "_matches.json")
        if not os.path.exists(t5_matchout) and not os.path.exists(esm2_matchout):
            with open(os.path.join(t5dir_disprot, filename)) as f:
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
                if os.path.exists(os.path.join(t5dir_disprot, "logits", uniprot_id + "_logits.pt")):
                    t5_dict['unmasked_logits'] = torch.load(os.path.join(t5dir_disprot, "logits", uniprot_id + "_logits.pt"))
                    T5_unmasked_aaloss_sequence   = get_single_aa_losses_indiv(t5_dict['unmasked_logits'].squeeze(), T5_prot_toks)
                else:
                    raise ValueError(f"Missing unmasked logits for {uniprot_id}")
                
                ## save T5_aaloss_sequences to a file
                os.makedirs(os.path.join(t5dir_disprot, "losses"), exist_ok=True)
                with open(os.path.join(t5dir_disprot, "losses", uniprot_id + "_losses.json"), "w") as f:
                    json.dump(T5_aaloss_sequences, f)

                ## save T5_unmasked_aaloss_sequence to a file
                with open(os.path.join(t5dir_disprot, "losses", uniprot_id + "_unmasked_losses.json"), "w") as f:
                    json.dump(T5_unmasked_aaloss_sequence, f)

                ## save matches to a file
                os.makedirs(os.path.join(t5dir_disprot, "matches"), exist_ok=True)
                with open(t5_matchout, "w") as f:
                    json.dump(t5_dict['aamask_1']['match'], f)

                if os.path.exists(os.path.join(esm2dir, uniprot_id + ".json")) and os.path.exists(os.path.join(esm2dir, "logits", uniprot_id + "_logits.pt")):
                    esm2_dict = json.load(open(os.path.join(esm2dir, uniprot_id + ".json")))[uniprot_id]
                    batch_labels, batch_strs, batch_tokens = batch_converter([(uniprot_id, input_seq)])
                    ESM2_prot_toks = batch_tokens.squeeze()[1:-1].numpy().tolist() ## remove start and end of sequence token
                    ESM2_aaloss_sequences = get_single_aa_losses(esm2_dict, ESM2_prot_toks)
                    esm2_dict['unmasked_logits'] = torch.load(os.path.join(esm2dir, "logits", uniprot_id + "_logits.pt"))
                    ESM2_unmasked_aaloss_sequence = get_single_aa_losses_indiv(esm2_dict['unmasked_logits'].squeeze(), ESM2_prot_toks)

                    ## save ESM2_aaloss_sequences to a file
                    os.makedirs(os.path.join(esm2dir, "losses"), exist_ok=True)
                    with open(os.path.join(esm2dir, "losses", uniprot_id + "_losses.json"), "w") as f:
                        json.dump(ESM2_aaloss_sequences, f)

                    ## save ESM2_unmasked_aaloss_sequence to a file
                    with open(os.path.join(esm2dir, "losses", uniprot_id + "_unmasked_losses.json"), "w") as f:
                        json.dump(ESM2_unmasked_aaloss_sequence, f)

                    ## save matches to a file
                    os.makedirs(os.path.join(esm2dir, "matches"), exist_ok=True)
                    with open(esm2_matchout, "w") as f:
                        json.dump(esm2_dict['aamask_1']['match'], f)
                else:
                    print(f"Missing ESM2 data for {uniprot_id}")
        else:
            print(f"Skipping {uniprot_id}")
