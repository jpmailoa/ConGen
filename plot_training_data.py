import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

def canonocalize(smi):
    
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

def combine_csv_files(csv_uri_list):

    def mini_load_csv(csv_uri):
        assert csv_name.endswith('.csv')
        data = pds.read_csv( csv_name )
        names = [name for name in data]
        assert names[0] == 'SMILES'
        smiles = []
        for j in range(len(data['SMILES'])):
            smiles.append(canonocalize(data['SMILES'][j]))
        data = data.to_numpy()
        if len(names)>1:
            Y_names = names[1:]
            Y_labels = np.asarray( data[:,1:], dtype=np.float32 )
        return smiles, Y_names, Y_labels
    
    n_csv = len(csv_uri_list)
    all_smiles = []
    all_Y_names = []
    all_Y_labels = []
    merged_smiles = []
    merged_Y_names = []
    smiles_dict = dict()
    for i in range(n_csv):
        csv_name = csv_uri_list[i]
        smiles, Y_names, Y_labels = mini_load_csv( csv_name )
        all_smiles.append( smiles )
        all_Y_names.append( Y_names )
        all_Y_labels.append( Y_labels )
        merged_smiles += smiles
        merged_Y_names += Y_names
    merged_smiles = list(set(merged_smiles))
    merged_Y_names = list(set(merged_Y_names))
    merged_Y_labels = np.zeros([len(merged_smiles), len(merged_Y_names)]).astype(float)
    merged_mask = np.zeros([len(merged_smiles), len(merged_Y_names)]).astype(float)
    count = 0
    for smiles in merged_smiles:
        smiles_dict[ smiles ] = count
        count += 1
    for i in range(n_csv):
        smiles_idx = [ smiles_dict[smiles] for smiles in all_smiles[i] ]
        Y_names_idx = [ merged_Y_names.index(Y_name) for Y_name in all_Y_names[i] ]
        temp_mask = np.ones([len(all_smiles[i]), len(all_Y_names[i])]).astype(float)
        temp_labels = np.copy(all_Y_labels[i])
        ## in case input is nan
        nan_idx = np.where( np.isnan(all_Y_labels[i]) )
        for j in range(len(nan_idx[0])):
            idx0 = nan_idx[0][j]
            idx1 = nan_idx[1][j]
            temp_mask[ idx0,idx1 ] = merged_mask[ smiles_idx[idx0],Y_names_idx[idx1] ]
            temp_labels[ idx0,idx1 ] = merged_Y_labels[ smiles_idx[idx0],Y_names_idx[idx1] ]
        ##
        merged_mask[ np.ix_(smiles_idx,Y_names_idx) ] = temp_mask
        merged_Y_labels[ np.ix_(smiles_idx,Y_names_idx) ] = temp_labels
    # Sort    
    seq = np.array( merged_Y_names ).argsort()
    merged_Y_names = [ merged_Y_names[i] for i in seq ]
    temp = merged_Y_labels[:, seq]
    merged_Y_labels = temp
    temp = merged_mask[:, seq]
    merged_mask = temp
    return merged_smiles, merged_Y_names, merged_Y_labels, merged_mask

data_uri = ['./data/paper_MP_IE_EA.csv',
            './data/paper_MP_clean_canonize_cut.csv',
            './data/paper_ZINC_310k.csv',
            './data/paper_clean_viscosity.csv',
            './data/paper_pubchem_fluorocarbon.csv']

smiles, Y_names, Y_labels, mask = combine_csv_files( data_uri )
idx_EA, idx_IE, idx_LogVis, idx_MolWt, idx_n_F, idx_n_O = [Y_names.index(prop) for prop in ['EA','IE','LogVis','MolWt','n_F','n_O']]

EA = [Y_labels[i,idx_EA] for i in range(Y_labels.shape[0]) if mask[i,idx_EA]==1.0]
IE = [Y_labels[i,idx_IE] for i in range(Y_labels.shape[0]) if mask[i,idx_IE]==1.0]
LogVis = [Y_labels[i,idx_LogVis] for i in range(Y_labels.shape[0]) if mask[i,idx_LogVis]==1.0]
MolWt = [Y_labels[i,idx_MolWt] for i in range(Y_labels.shape[0]) if mask[i,idx_MolWt]==1.0]
n_F = [Y_labels[i,idx_n_F] for i in range(Y_labels.shape[0]) if mask[i,idx_n_F]==1.0]
n_O = [Y_labels[i,idx_n_O] for i in range(Y_labels.shape[0]) if mask[i,idx_n_O]==1.0]

print('Total molecule:', len(smiles))
color = 'darkgoldenrod'

# EA
plt.figure(0, figsize=[5,5])
plt.hist(EA, bins=[i*0.1 for i in range(-20,71)], color=color)
plt.xlabel(r'EA (eV)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([0,0], [0,4000], 'k--')
plt.plot([0.5,0.5], [0,4000], 'k--')
plt.axis([-2,7,0,4000])
plt.tight_layout()
# IE
plt.figure(1, figsize=[5,5])
plt.hist(IE, bins=[i*0.1 for i in range(0,101)], color=color)
plt.xlabel(r'IE (eV)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([7.0,7.0], [0,3000], 'k--')
plt.plot([7.5,7.5], [0,3000], 'k--')
plt.axis([0,10,0,3000])
plt.tight_layout()
# LogVis
plt.figure(2, figsize=[5,5])
plt.hist(LogVis, bins=[i*0.1 for i in range(-7,18)], color=color)
plt.xlabel(r'Log.Vis', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([0,0], [0,40], 'k--')
plt.plot([-0.1,-0.1], [0,40], 'k--')
plt.axis([-0.7,1.8,0,40])
plt.tight_layout()
# MolWt
plt.figure(3, figsize=[5,5])
plt.hist(MolWt, bins=[i*10 for i in range(0,51)], color=color)
plt.xlabel(r'Mol.Wt (Da)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([300,300], [0,30000], 'k--')
plt.plot([250,250], [0,30000], 'k--')
plt.axis([0,500,0,30000])
plt.tight_layout()
# n_F
plt.figure(4, figsize=[5,5])
plt.hist(n_F, bins=[i-0.5 for i in range(0,13)], color=color)
plt.xlabel(r'$n_F$', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([4,4], [0,350000], 'k--')
plt.plot([6,6], [0,350000], 'k--')
plt.axis([0,13,0,350000])
plt.tight_layout()
# n_O
plt.figure(5, figsize=[5,5])
plt.hist(n_O, bins=[i-0.5 for i in range(0,11)], color=color)
plt.xlabel(r'$n_O$', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([2,2], [0,200000], 'k--')
plt.plot([1,1], [0,200000], 'k--')
plt.axis([0,10,0,200000])
plt.tight_layout()

plt.show()
