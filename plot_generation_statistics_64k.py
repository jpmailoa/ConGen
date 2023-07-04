import matplotlib.pyplot as plt
import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import pandas as pds
from preprocessing import canonocalize
import json

def load_curves(ipynb_file):
    cond = dict()
    data = json.load(open(ipynb_file,'r'))
    data = data['cells']
    gen_cond_blocks = data[12]['outputs']
    count = 0
    for block in gen_cond_blocks:
        for line in block['text']:
            words = line.strip().split()
            if len(words)==5:
                idx = count #int(words[0][1:-1])
                smiles = str(words[1][1:-2])
                prop1 = float(words[2][1:-1])
                prop2 = float(words[3][0:-1])
                prop3 = float(words[4][0:-2])
                cond[ count ] = [smiles, [prop1,prop2,prop3] ]
                count += 1
            
    return cond

def sampling_statistics(dict_out):
    out = []
    with open('Conditional_Query2.csv','w') as f:
        f.write('SMILES,MolWt,nF,nO\n')
        for idx, [smiles, prop] in dict_out.items():
            mol = Chem.MolFromSmiles(smiles)
            MolWt = Descriptors.ExactMolWt(mol)
            nF, nO = 0, 0
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol == 'F': nF += 1
                if symbol == 'O': nO += 1
            prop = [MolWt, nF, nO]
            f.write(','.join([str(x) for x in [smiles]+prop])+'\n')
            out.append(prop)
    out = np.array(out)
    return np.mean(out,axis=0), np.std(out,axis=0), out

def average(in_list):
    y = []
    for item in in_list:
        y_avg = np.array(item)
        y_avg = np.mean(y_avg)
        y.append(y_avg)
    y = np.array(y)
    return y

def check_charge(list_out):
    out = []
    for smiles in list_out:
        charge = 0
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            charge += atom.GetFormalCharge()
        out.append(charge)
    out = np.array(out)
    return out

cond_ori = load_curves('repeat_generation_64k.ipynb')

fts = 18

# Conditional sampling
s_ori = sampling_statistics(cond_ori)
print('Conditional sampling mean Original RNN:\t',s_ori[0])
print('Conditional sampling std Original RNN:\t',s_ori[1])

# Conditional generation distribution
start = 7
# MolWt
plt.figure(start, figsize=[5,5])
plt.hist(s_ori[2][:,0], bins=[i*10 for i in range(10,46)], color='seagreen', alpha=0.5)
plt.plot([250,250],[0,20000],'k--')
plt.plot([300,300],[0,20000],'k--')
plt.xlabel('Mol.Wt (Da)', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([100,450,0,20000])
plt.tight_layout()
# nF
plt.figure(start+1, figsize=[5,5])
plt.hist(s_ori[2][:,1], bins=[i-0.5 for i in range(0,8)], color='seagreen', alpha=0.5)
plt.plot([4,4],[0,40000],'k--')
plt.plot([6,6],[0,40000],'k--')
plt.xlabel(r'$n_F$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([0,8,0,40000])
plt.tight_layout()
# nO
plt.figure(start+2, figsize=[5,5])
plt.hist(s_ori[2][:,2], bins=[i-0.5 for i in range(0,6)], color='seagreen', alpha=0.5)
plt.plot([1,1],[0,40000],'k--')
plt.plot([2,2],[0,40000],'k--')
plt.xlabel(r'$n_O$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([0,6,0,40000])
plt.tight_layout()

csv_files = ['./data/paper_MP_IE_EA.csv',
             './data/paper_MP_clean_canonize_cut.csv',
             './data/paper_ZINC_310k.csv',
             './data/paper_clean_viscosity.csv',
             './data/paper_pubchem_fluorocarbon.csv']
training_smiles = []
for csv_file in csv_files:
    data = pds.read_csv( csv_file )
    for i in range(len(data['SMILES'])):
        training_smiles.append( canonocalize(data['SMILES'][i]) )
training_smiles = list(set(training_smiles))
canon_cond_ori_smiles = [canonocalize(cond_ori[idx][0]) for idx in cond_ori]
unique_cond_ori_smiles = list(set(canon_cond_ori_smiles))
not_train_cond_ori_smiles = [smiles for smiles in unique_cond_ori_smiles if not (smiles in training_smiles)]
print('Invalid SMILES count:', (2**6 *1000) - len(cond_ori))
print('Valid SMILES count:', len(cond_ori))
print('Unique SMILES count:', len(unique_cond_ori_smiles))
print('New SMILES count:', len(not_train_cond_ori_smiles))

plt.show()
