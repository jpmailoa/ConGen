# pip install transformers
# Dependency:
import pickle
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel
import argparse
from rdkit import Chem

def canonocalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

def main(args):
    if torch.cuda.is_available(): dev = torch.device('cuda:0')
    else: dev = torch.device('cpu')

    print('Downloading neural model.')
    model = RobertaModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(dev)
    tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

    # Load data
    data = pd.read_csv(args.csv_file)
    print('Loading data of size: {}'.format(data.shape))

    # Load existing nn_fps
    if args.input_vocab_file:
        with open(args.input_vocab_file, 'rb') as f:
            smiles2nn_fps = pickle.load(f)
        print('Loading existing neural fingerprints.')
    else:
        smiles2nn_fps = {}

    # Build latest nn_fps
    smiles = [canonocalize(entry) for entry in list(set(data['SMILES'].to_list()))]
    print('Building latest neural fingerprints...')
    for smi in tqdm(smiles):
        if smi in smiles2nn_fps and args.fp_check == 'False':
            continue

        inputs = tokenizer.encode_plus(smi, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids'].to(dev)

        with torch.no_grad():
            nn_fp = model(input_ids)['pooler_output'].view(-1).detach().cpu().numpy()
        if smi not in smiles2nn_fps:
            smiles2nn_fps[smi] = nn_fp
        else: # check if the nn_fp is consistent
            assert np.array_equal(smiles2nn_fps[smi], nn_fp) 

    # Write to pickle file
    with open(args.output_vocab_file, 'wb') as f:
        pickle.dump(smiles2nn_fps, f)
    print('Write to {}.'.format(args.output_vocab_file))

##if __name__ == '__main__':
##    parser = argparse.ArgumentParser()
##    parser.add_argument('--csv_file', type=str, help='path of the csv file')
##    parser.add_argument('--input_vocab_file', type=str, default='', help='path of existing vocab file')
##    parser.add_argument('--output_vocab_file', type=str, help='path of the output vocab file')
##    parser.add_argument('--fp_check', type=str, default='True', choices=['True', 'False'], help='check whether the new fp is the same as the previous one')
##    args = parser.parse_args()
##    main(args)
