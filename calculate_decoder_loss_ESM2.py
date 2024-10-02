from tqdm import tqdm
import random
import os
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
import re
import esm
import torch
import json

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

    opts = parser.parse_args()
    return opts

def sequence_masker(seq, i, j):
    masked_sequence_list = seq.split()
    if j<=i:
        print(f"index j={j} must be greater than i={i}")
        raise
    for x in range(i, j):
        if j > len(seq):
            break
        masked_sequence_list[x] = f"<mask>"
    return " ".join(masked_sequence_list)

if __name__ == "__main__": 

    opts = parse_args()

    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("Parameters:")
    print(f"\t- Output attentions: {opts.output_attentions}")

    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)
    if not os.path.exists(opts.outdir+"/logits"):
        os.makedirs(opts.outdir+"/logits")
    if opts.output_attentions:
        if not os.path.exists(opts.outdir+"/attentions"):
            os.makedirs(opts.outdir+"/attentions")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\t- Device: {device}")
    # Load model and tokenizer
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

    if device.type == 'cuda':
        model = model.eval().cuda()  # disables dropout for deterministic results
        torch.cuda.empty_cache()
    else:
        model = model.eval()

    # Define the loss function obj
    loss = torch.nn.CrossEntropyLoss()

    # Read fasta sequences
    counter = 0
    batch_converter = alphabet.get_batch_converter()
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
            batch_labels, batch_strs, batch_tokens = batch_converter([(uniprot_id, input_seq[0])])
            if not os.path.exists(f"{opts.outdir}/{uniprot_id}.json"):
                for mask_size in mask_sizes:
                    print(f"#### Mask size: {mask_size} ####")
                        
                    loss_sequence = list()
                    match_sequence = list()
                    logits_sequence = list()
                    for i in tqdm(range(len(target_seq)-mask_size+1)):

                        masked_seq = sequence_masker(input_seq[0], i, i+mask_size)
                        mbatch_labels, mbatch_strs, mbatch_tokens = batch_converter([(uniprot_id, masked_seq)])
                        with torch.no_grad():
                            results = model(mbatch_tokens.to(device), repr_layers=[36], return_contacts=False)
                        cpulogits = results['logits'][0].cpu()
                        loss_val = float(loss(cpulogits[1:-1,], mbatch_tokens[0][1:-1]).numpy())  ## recently corrected to discard cls and eos tokens
                        loss_sequence.append(loss_val)
                        logits_sequence.append(cpulogits[1:-1,].numpy().tolist())
                        #fastpred = tokenizer.decode(torch.tensor(cpulogits[:,:-1,:].numpy().argmax(-1)[0]), skip_special_tokens=False).replace("<"," <").replace(">","> ")
                        fastpred = " ".join([alphabet.get_tok(t) for t in results['logits'][0].cpu().numpy().argmax(-1)][1:-1]) ## delete first and last tokens
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
                    ### This takes too much time and space, we will save the logits somewhere else
                    #pred_dict[uniprot_id][f"aamask_{mask_size}"]["logits"] = logits_sequence
                    np.save(f"{opts.outdir}/logits/{uniprot_id}_logits_sequence.npy",  np.array(logits_sequence, dtype=object), allow_pickle=True)
                with open(f"{opts.outdir}/{uniprot_id}.json", 'w') as outfmt:
                    json.dump(pred_dict, outfmt)
            else:
                print(f"Skipping {uniprot_id} masks")
            if not os.path.exists(f"{opts.outdir}/logits/{uniprot_id}_logits.pt"):
                ## Output the complete attention matrices with a full pass, no mask
                with torch.no_grad():
                    #emb = fullmodel(input_ids=true_tok, labels=true_tok, output_attentions=opts.output_attentions, attention_mask=attention_mask, decoder_attention_mask=attention_mask)
                    if device.type == 'cuda':
                        # if len(aa_sequence) > 600:
                        #     results = model(batch_tokens.to(device), repr_layers=[36], return_contacts=False)
                        # else:
                        results = model(batch_tokens.to(device), repr_layers=[36], return_contacts=False)
                        # torch.save(results['contacts'][0].cpu(), f"{opts.outdir}/logits/{uniprot_id}_contacts.pt")
                    else:
                        results = model(batch_tokens.to(device), repr_layers=[36], return_contacts=True)
                        torch.save(results['contacts'][0].cpu(), f"{opts.outdir}/logits/{uniprot_id}_contacts.pt")
                cpulogits = results['logits'][0].cpu()[1:-1,]
                torch.save(cpulogits, f"{opts.outdir}/logits/{uniprot_id}_logits.pt")
                if opts.output_attentions:
                    torch.save(results['attentions'][0], f"{opts.outdir}/attentions/{uniprot_id}_encoder_attentions.pt")
            else:
                print(f"Skipping {uniprot_id} attentions matrices and logits")
        counter += 1
        if device == 'cuda:0':
            torch.cuda.empty_cache()
