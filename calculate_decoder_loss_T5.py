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
from torch.nn import CrossEntropyLoss

def parse_args():

    parser = argparse.ArgumentParser("Calculate decoder loss and attention matrices for masked sequences")

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

    parser.add_argument("--outattentions",
                          action='store_true',
                          dest="output_attentions",
                          default=False,
                          help="Output attention matrices")
    
    parser.add_argument("--autoregressive",
                          action='store_true',
                          dest="autoregressive",
                          default=False,
                          help="generate output using autoregressive decoding")

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

    print("Parameters:")
    print(f"\t- Output attentions: {opts.output_attentions}")

    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)

    modelpath = "models/prottrans_t5_xl_u50/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    fullmodel = T5ForConditionalGeneration.from_pretrained(modelpath).to(device)
    celoss = CrossEntropyLoss()

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(modelpath, do_lower_case=False, legacy=False)
    torch.cuda.empty_cache()

    os.makedirs(f"{opts.outdir}/attentions", exist_ok=True)
    os.makedirs(f"{opts.outdir}/logits", exist_ok=True)
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
                    logits_sequence = list()
                    for i in tqdm(range(len(target_seq)-mask_size+1)):
                        masked_seq = [sequence_masker(input_seq[0], i, i+mask_size)]
                        tmp = tokenizer(masked_seq)
                        input_ids = torch.tensor(tmp['input_ids']).to(device)
                        
                        if opts.autoregressive:
                            with torch.no_grad():
                                emb = fullmodel.generate(input_ids, max_length=len(input_ids[0])+1, return_dict_in_generate=True, output_scores=True, output_attentions=opts.output_attentions)
                            cpulogits = torch.stack([s.squeeze() for s in emb.scores], dim=0)[:-1,].cpu()
                            fastpred = tokenizer.decode(torch.argmax(cpulogits, dim=1), skip_special_tokens=False).replace("<"," <").replace(">","> ")
                            
                            # Calculate protein-level loss from single aminoacid losses
                            acum = 0
                            for j in range(cpulogits.shape[0]):
                                acum += celoss(cpulogits[j], true_tok.cpu().squeeze()[:-1][j])        
                            protein_loss  = acum/cpulogits.shape[0]
                            # the proper full protein loss should be the one below, but it will fail if dimentions are not correct
                            # the above expression will be wrong, but aproximate? maybe shifted by one? it will suck, let's set it to -1
                            ## protein_loss2 = celoss(cpulogits, true_tok.squeeze()[:-1])
                                
                        else:
                            with torch.no_grad():
                                emb = fullmodel(input_ids=input_ids, labels=true_tok, output_attentions=opts.output_attentions)
                            protein_loss = emb.loss.cpu()
                            
                            cpulogits = emb.logits.cpu().squeeze()[:-1]
                            ## fastpred = tokenizer.decode(torch.tensor(cpulogits[:,:-1,:].numpy().argmax(-1)[0]), skip_special_tokens=False).replace("<"," <").replace(">","> ")
                            fastpred = tokenizer.decode(torch.argmax(cpulogits, dim=1), skip_special_tokens=False).replace("<"," <").replace(">","> ")
                            
                        logits_sequence.append(cpulogits.numpy().tolist())
                        if input_seq[0] == fastpred:
                            match_sequence.append(True)
                            loss_sequence.append(protein_loss.item())
                        else:
                            pred_arr = fastpred.split()
                            seq_arr  = input_seq[0].split()
                            if len(pred_arr) == len(seq_arr):
                                local_match_sequence = list()
                                for j in range(len(pred_arr)):
                                    if pred_arr[j] != seq_arr[j]:
                                        local_match_sequence.append((j,pred_arr[j], seq_arr[j]))
                                loss_sequence.append(protein_loss.item())
                                match_sequence.append(local_match_sequence)
                            else:
                                print(f"{i} - Mismatch length error")
                                match_sequence.append(False)
                                loss_sequence.append(-1)
                            
                            
                    pred_dict[uniprot_id][f"aamask_{mask_size}"] = dict()
                    pred_dict[uniprot_id][f"aamask_{mask_size}"]["match"] = match_sequence
                    pred_dict[uniprot_id][f"aamask_{mask_size}"]["loss"] = loss_sequence
                    ### This takes too much time and space, we will save the logits somewhere else
                    # pred_dict[uniprot_id][f"aamask_{mask_size}"]["logits"] = logits_sequence
                    np.save(f"{opts.outdir}/logits/{uniprot_id}_logits_sequence.npy",  np.array(logits_sequence, dtype=object), allow_pickle=True)
                with open(f"{opts.outdir}/{uniprot_id}.json", 'w') as outfmt:
                    json.dump(pred_dict, outfmt)
            else:
                print(f"Skipping {uniprot_id} masks")
            if not os.path.exists(f"{opts.outdir}/logits/{uniprot_id}_logits.pt"):
                ## Output the complete attention matrices with a full pass, no masked aminoacids
                with torch.no_grad():
                    if opts.autoregressive:
                        emb = fullmodel.generate(true_tok, max_length=len(true_tok[0])+1, return_dict_in_generate=True, output_scores=True, output_attentions=opts.output_attentions)
                        cpulogits = torch.stack([s.squeeze() for s in emb.scores], dim=0)[:-1,].cpu()
                        torch.save(cpulogits, f"{opts.outdir}/logits/{uniprot_id}_logits.pt")
                    else:
                        emb = fullmodel(input_ids=true_tok, labels=true_tok, output_attentions=opts.output_attentions)
                        cpulogits = emb.logits.squeeze()[:-1].cpu()
                        torch.save(cpulogits, f"{opts.outdir}/logits/{uniprot_id}_logits.pt")
                if opts.output_attentions:
                    os.makedirs(f"{opts.outdir}/attentions", exist_ok=True)
                    torch.save(emb.encoder_attentions, f"{opts.outdir}/attentions/{uniprot_id}_encoder_attentions.pt")
                    torch.save(emb.decoder_attentions, f"{opts.outdir}/attentions/{uniprot_id}_decoder_attentions.pt")
            else:
                print(f"Skipping {uniprot_id} attentions matrices and logits")
        counter += 1