


from tqdm import tqdm
import random
import os
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json

def parse_args():

    parser = argparse.ArgumentParser("Calculate decoder loss and attention matrices for masked sequences")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        help="choose prott5 or esm1/2 models. Needs full directory path (e.g models/prottrans_t5_xl_u50/)")

    parser.add_argument("--fasta",
                          type=str,
                          default=None,
                          dest="fastafile",
                          help="Multi fasta file")

    parser.add_argument("--start",
                          type=int,
                          default=0,
                          dest="start",
                          help="start at sequence nº")
    
    parser.add_argument("--end",
                          type=int,
                          default=1e8,
                          dest="end",
                          help="end at sequence nº")

    parser.add_argument("--outdir",
                          type=str,
                          dest="outdir",
                          help="Output directory")

    opts = parser.parse_args()
    return opts

def sequence_masker(seq, i, j, same_extra_token=False):
    masked_sequence_list = seq.split()
    token_num = 0
    if j<=i:
        print(f"index j={j} must be greater than i={i}")
        raise
    for x in range(i, j):
        if j > len(seq):
            break
        masked_sequence_list[x] = f"<extra_id_{token_num}>"
        if not same_extra_token:
            token_num += 1
    return " ".join(masked_sequence_list)

if __name__ == "__main__": 


    opts = parse_args()

    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fullmodel = T5ForConditionalGeneration.from_pretrained(opts.model).to(device)

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(opts.model, do_lower_case=False, legacy=False)
    torch.cuda.empty_cache()

    # Read fasta sequences
    counter = 0
    for record in SeqIO.parse(opts.fastafile, "fasta"):
        #sequences.append(record)
        if counter >= opts.start and counter < opts.end:
            uniprot_id = record.id
            aa_sequence = str(record.seq)
            
            pred_dict = dict()
            mask_sizes = [1]
            print(f" Processing {uniprot_id}, protnum {counter}")
            if uniprot_id not in pred_dict:
                pred_dict[uniprot_id] = dict()
            
            target_seq = aa_sequence
            input_seq = [" ".join(list(re.sub(r"[UZOB]", "X", target_seq)))]
            true_input = tokenizer(input_seq)
            true_tok = torch.tensor(true_input['input_ids']).to(device)
            attention_mask = torch.tensor(true_input['attention_mask']).to(device)
            if not os.path.exists(f"{opts.outdir}/{uniprot_id}.json"):
                for mask_size in mask_sizes:
                    print(f"#### Mask size: {mask_size} ####")
                        
                    loss_sequence = list()
                    match_sequence = list()
                    for i in tqdm(range(len(target_seq)-mask_size+1)):

                        masked_seq = [sequence_masker(input_seq[0], i, i+mask_size)]
                        tmp = tokenizer(masked_seq)
                        input_ids = torch.tensor(tmp['input_ids']).to(device)
                        attention_mask = torch.tensor(tmp['attention_mask']).to(device)
                        with torch.no_grad():
                            emb  = fullmodel(input_ids=input_ids, labels=true_tok, output_attentions=True, attention_mask=attention_mask, decoder_attention_mask=attention_mask)
                            loss = emb.loss.cpu()
                            loss_sequence.append(loss.item())
                            cpulogits = emb.logits.cpu()
                            fastpred = tokenizer.decode(torch.tensor(cpulogits[:,:-1,:].numpy().argmax(-1)[0]), skip_special_tokens=False).replace("<"," <").replace(">","> ")
                            if input_seq[0] == fastpred:
                                match_sequence.append(True)
                            else:
                                pred_arr = fastpred.split()
                                seq_arr  = input_seq[0].split()
                                if len(pred_arr) == len(seq_arr):
                                    local_match_sequence = list()
                                    for j in range(len(pred_arr)):
                                        if pred_arr[j] != seq_arr[j]:
                                            local_match_sequence.append((j,pred_arr[j], seq_arr[j]))
                                else:
                                    print("Mismatch length error")
                                    raise
                                match_sequence.append(local_match_sequence)
                            
                    pred_dict[uniprot_id][f"aamask_{mask_size}"] = dict()
                    pred_dict[uniprot_id][f"aamask_{mask_size}"]["match"] = match_sequence
                    pred_dict[uniprot_id][f"aamask_{mask_size}"]["loss"] = loss_sequence
                with open(f"{opts.outdir}/{uniprot_id}.json", 'w') as outfmt:
                    json.dump(pred_dict, outfmt)
            else:
                print(f"Skipping {uniprot_id} masks")
            if not os.path.exists(f"logits/{uniprot_id}_logits.pt"):
                ## Output the complete attention matrices with a full pass, no mask
                with torch.no_grad():
                    emb = fullmodel(input_ids=true_tok, labels=true_tok, output_attentions=True, attention_mask=attention_mask, decoder_attention_mask=attention_mask)
                    torch.save(emb.encoder_attentions, f"attentions/{uniprot_id}_encoder_attentions.pt")
                    torch.save(emb.decoder_attentions, f"attentions/{uniprot_id}_decoder_attentions.pt")
                    torch.save(emb.logits, f"logits/{uniprot_id}_logits.pt")
            else:
                print(f"Skipping {uniprot_id} attentions matrices and logits")
        counter += 1