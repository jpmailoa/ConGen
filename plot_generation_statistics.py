import matplotlib.pyplot as plt
import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import pandas as pds
from preprocessing import canonocalize

def EA_IE_statistics(csv_file):
    data = pds.read_csv(csv_file)
    EA = np.array([i for i in data['EA'].to_numpy() if not np.isnan(i)])
    IE = np.array([i for i in data['IE'].to_numpy() if not np.isnan(i)])
    return [EA.mean(),IE.mean()], [EA.std(),IE.std()], data[['EA','IE']].to_numpy()

def load_curves(log_file):
    trn, val = [], []
    trn_LU, trn_Y = [], []
    val_LU, val_Y = [], []
    temp_trn_LU, temp_trn_Y = [], []
    temp_val_LU, temp_val_Y = [], []
    test = []
    uncond, cond = [], []
    with open(log_file,'r') as f:
        line = f.readline().strip()
        while line:
            if "['Training', 'cost_trn'" in line:
                cost = float(line.split()[-1][:-1])
                trn.append(cost)
                trn_LU.append(temp_trn_LU)
                trn_Y.append(temp_trn_Y)
                temp_trn_LU, temp_trn_Y = [], []
            elif "['Validation', 'cost_val'" in line:
                cost = float(line.split()[-1][:-1])
                val.append(cost)
                val_LU.append(temp_val_LU)
                val_Y.append(temp_val_Y)
                temp_val_LU, temp_val_Y = [], []
            elif "Training Cycle" in line:
                temp_cycle_LU = temp_trn_LU
                temp_cycle_Y = temp_trn_Y
            elif "Validation Cycle" in line:
                temp_cycle_LU = temp_val_LU
                temp_cycle_Y = temp_val_Y
            elif "--Cost Train LU" in line:
                cost = float(line.split()[-1])
                temp_cycle_LU.append(cost)
            elif "--Cost Train Y" in line:
                cost = float(line.split()[-1])
                temp_cycle_Y.append(cost)
            elif "Unconditional Generation" in line:
                to_use = uncond
            elif "Conditional Generation" in line:
                to_use = cond
            elif line.startswith('[') and line.endswith(']'):
                words = line.split()
                if len(words)==2:
                    MAE = float(words[-1][:-1])
                    test.append(MAE)
                elif len(words)==5:
                    smiles = str(words[1][1:-2])
                    to_use.append( smiles )
            line = f.readline().strip()
            # to prevent bug related to RDKit error message, we require
            # 2 continuous empty lines before we stop reading the log file
            if not line:
                line = f.readline().strip()
    return trn, trn_LU, trn_Y, val, val_LU, val_Y, test, uncond, cond

def sampling_statistics(list_out):
    out = []
    for smiles in list_out:
        mol = Chem.MolFromSmiles(smiles)
        MolWt = Descriptors.ExactMolWt(mol)
        nF, nO = 0, 0
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'F': nF += 1
            if symbol == 'O': nO += 1
        prop = [MolWt, nF, nO]
        print(','.join([str(x) for x in [smiles]+prop]))
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

trn_ori, trn_ori_LU, trn_ori_Y, val_ori, val_ori_LU, val_ori_Y, test_ori, uncond_ori, cond_ori = load_curves('models/221122121322/status.log')

### Training
##plt.figure(0, figsize=[5,5])
##plt.semilogy(trn_ori, 'b', linewidth=2)
##plt.xlabel('Iteration', fontsize=14)
##plt.ylabel('Cost', fontsize=14)
##plt.xticks(fontsize=14)
##plt.yticks(fontsize=14)
##plt.title('Training', fontsize=14)
##plt.legend(['Original RNN'], fontsize=14)
##plt.tight_layout()
##
### Validation
##plt.figure(1, figsize=[5,5])
##plt.semilogy(val_ori, 'b', linewidth=2)
##plt.xlabel('Iteration', fontsize=14)
##plt.ylabel('Cost', fontsize=14)
##plt.xticks(fontsize=14)
##plt.yticks(fontsize=14)
##plt.title('Validation', fontsize=14)
##plt.legend(['Original RNN'], fontsize=14)
##plt.tight_layout()
##
### Training LU
##plt.figure(2, figsize=[5,5])
##plt.semilogy(average(trn_ori_LU), 'b', linewidth=2)
##plt.xlabel('Iteration', fontsize=14)
##plt.ylabel('Cost', fontsize=14)
##plt.xticks(fontsize=14)
##plt.yticks(fontsize=14)
##plt.title('Training LU', fontsize=14)
##plt.legend(['Original RNN'], fontsize=14)
##plt.tight_layout()
##
### Training Y
##plt.figure(3, figsize=[5,5])
##plt.semilogy(average(trn_ori_Y), 'b', linewidth=2)
##plt.xlabel('Iteration', fontsize=14)
##plt.ylabel('Cost', fontsize=14)
##plt.xticks(fontsize=14)
##plt.yticks(fontsize=14)
##plt.title('Training Y', fontsize=14)
##plt.legend(['Original RNN'], fontsize=14)
##plt.tight_layout()

# Test
print('Test MAE Original RNN:\t',test_ori)

fts = 18

# Unconditional sampling
print('Unconditional')
s_ori = sampling_statistics(uncond_ori)
qc_ori = EA_IE_statistics('models/221122121322/Unconditional.csv')
print('Unconditional sampling mean Original RNN:\t',s_ori[0],qc_ori[0])
print('Unconditional sampling std Original RNN:\t',s_ori[1],qc_ori[1])

# Unconditional generation distribution
start = 0
# MolWt
plt.figure(start, figsize=[5,5])
plt.hist(s_ori[2][:,0], bins=[i*10 for i in range(10,46)], color='red', alpha=0.5)
plt.xlabel('Mol.Wt (Da)', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([100,450,0,5])
plt.tight_layout()
# nF
plt.figure(start+1, figsize=[5,5])
plt.hist(s_ori[2][:,1], bins=[i-0.5 for i in range(0,8)], color='red', alpha=0.5)
plt.xlabel(r'$n_F$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([0,8,0,5])
plt.tight_layout()
# nO
plt.figure(start+2, figsize=[5,5])
plt.hist(s_ori[2][:,2], bins=[i-0.5 for i in range(0,6)], color='red', alpha=0.5)
plt.xlabel(r'$n_O$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([0,6,0,5])
plt.tight_layout()
# EA
plt.figure(start+3, figsize=[5,5])
plt.hist(qc_ori[2][:,0], bins=[i*0.1 for i in range(-10,40)], color='red', alpha=0.5)
plt.xlabel(r'$EA$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([-1,4,0,5])
plt.tight_layout()
# IE
plt.figure(start+4, figsize=[5,5])
plt.hist(qc_ori[2][:,1], bins=[i*0.1 for i in range(40,90)], color='red', alpha=0.5)
plt.xlabel(r'$IE$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([4,9,0,5])
plt.tight_layout()

# Conditional sampling
print('Conditional')
s_ori = sampling_statistics(cond_ori)
qc_ori = EA_IE_statistics('models/221122121322/Conditional_Query1.csv')
print('Conditional sampling mean Original RNN:\t',s_ori[0],qc_ori[0])
print('Conditional sampling std Original RNN:\t',s_ori[1],qc_ori[1])

# Unconditional generation distribution
start = 5
# MolWt
plt.figure(start, figsize=[5,5])
plt.hist(s_ori[2][:,0], bins=[i*10 for i in range(10,46)], color='blue', alpha=0.5)
plt.plot([250,250],[0,100],'k--')
plt.plot([300,300],[0,100],'k--')
plt.xlabel('Mol.Wt (Da)', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([100,450,0,100])
plt.tight_layout()
# nF
plt.figure(start+1, figsize=[5,5])
plt.hist(s_ori[2][:,1], bins=[i-0.5 for i in range(0,8)], color='blue', alpha=0.5)
plt.plot([4,4],[0,200],'k--')
plt.plot([6,6],[0,200],'k--')
plt.xlabel(r'$n_F$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([0,8,0,200])
plt.tight_layout()
# nO
plt.figure(start+2, figsize=[5,5])
plt.hist(s_ori[2][:,2], bins=[i-0.5 for i in range(0,6)], color='blue', alpha=0.5)
plt.plot([1,1],[0,200],'k--')
plt.plot([2,2],[0,200],'k--')
plt.xlabel(r'$n_O$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([0,6,0,200])
plt.tight_layout()
# EA
plt.figure(start+3, figsize=[5,5])
plt.hist(qc_ori[2][:,0], bins=[i*0.1 for i in range(-10,40)], color='blue', alpha=0.5)
plt.plot([0,0],[0,35],'k--')
plt.plot([0.5,0.5],[0,35],'k--')
plt.xlabel(r'$EA$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([-1,4,0,35])
plt.tight_layout()
# IE
plt.figure(start+4, figsize=[5,5])
plt.hist(qc_ori[2][:,1], bins=[i*0.1 for i in range(40,90)], color='blue', alpha=0.5)
plt.plot([7.0,7.0],[0,35],'k--')
plt.plot([7.5,7.5],[0,35],'k--')
plt.xlabel(r'$IE$', fontsize=fts)
plt.ylabel('Count', fontsize=fts)
plt.xticks(fontsize=fts)
plt.yticks(fontsize=fts)
plt.axis([4,9,0,35])
plt.tight_layout()


##csv_files = ['./data/paper_MP_IE_EA.csv',           
##                   './data/paper_MP_clean_canonize_cut.csv',
##                   './data/paper_ZINC_310k.csv',
##                   './data/paper_clean_viscosity.csv',
##                   './data/paper_pubchem_fluorocarbon.csv']
##training_smiles = []
##for csv_file in csv_files:
##    data = pds.read_csv( csv_file )
##    for i in range(len(data['SMILES'])):
##        training_smiles.append( canonocalize(data['SMILES'][i]) )
##training_smiles = list(set(training_smiles))
##canon_cond_ori_smiles = [canonocalize(smiles) for smiles in cond_ori]
##unique_cond_ori_smiles = list(set(canon_cond_ori_smiles))
##not_train_cond_ori_smiles = [smiles for smiles in unique_cond_ori_smiles if not (smiles in training_smiles)]
##print('Invalid SMILES count:', (2**6 *5) - len(cond_ori))
##print('Valid SMILES count:', len(cond_ori))
##print('Unique SMILES count:', len(unique_cond_ori_smiles))
##print('New SMILES count:', len(not_train_cond_ori_smiles))

plt.show()
