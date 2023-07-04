import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import pickle
import pandas as pds
from sklearn.preprocessing import StandardScaler
from build_nnfps import main as nnfps_main

def get_property(smi):

    try:
        mol=Chem.MolFromSmiles(smi) 
        property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), QED.qed(mol)]
        
    except:
        property = 'invalid'
           
    return property
    

def canonocalize(smi):

    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def vectorize(list_input, char_set):

    one_hot = np.zeros((list_input.shape[0], list_input.shape[1]+4, len(char_set)), dtype=np.int32)

    for si, ss in enumerate(list_input):
        for cj, cc in enumerate(ss):
            one_hot[si,cj+1,cc] = 1

        one_hot[si,-1,0] = 1
        one_hot[si,-2,0] = 1
        one_hot[si,-3,0] = 1

    return one_hot[:,0:-1,:], one_hot[:,1:,:]


def smiles_to_seq(smiles, char_set):

    char_to_int = dict((c,i) for i,c in enumerate(char_set))
    
    list_seq=[]
    for s in smiles:
        seq=[]                
        j=0
        while j<len(s):
            if j<len(s)-1 and s[j:j+2] in char_set:
                seq.append(char_to_int[s[j:j+2]])
                j=j+2
    
            elif s[j] in char_set:
                seq.append(char_to_int[s[j]])
                j=j+1
            else:
                raise Exception('Unrecognized character '+(s[j])+' not in known SMILES character set, SMILES: '+s)
    
        list_seq.append(seq)
    ori_list_seq = list_seq
    list_seq = np.zeros([len(list_seq), max([len(i) for i in ori_list_seq])], dtype=int)
    for i in range(len(ori_list_seq)):
        list_seq[i,0:len(ori_list_seq[i])] = ori_list_seq[i]
    
    return list_seq


def smiles_pretrain(smiles_list, dict_path):
    dict_out = pickle.load(open(dict_path,'rb'))
    out = np.array( [ dict_out[ smiles ] for smiles in smiles_list ] )
    return out

def _char_set():
    char_set = [' ','0','1','2','3','4','5','6','7','8','9','-','#','(',')','[',']','+','=','b','B','Br','c','C','Cl','F','H','I','N','n','O','o','P','p','S','s','Si','si','Sn','@','/','\\','%']
    return char_set

def _corrupt_training_data(Y):
    corrupt_fraq = 0.50
    shape = Y.shape
    length = len(Y.reshape([-1]))
    mask = np.random.rand( length ).reshape(shape)
    mask = ( mask > corrupt_fraq ).astype(float)
    corrupt_Y = Y * mask
    return corrupt_Y, mask

def data_preparation(data_uri, ntrn, ntst,
                     frac_val = 0.05,     # fraction of labeled/unlabeled training data to be used as validation set
                     pretrain_uri = None,
                     temp_csv = 'data/temp.csv'):
    if type(data_uri) is str:
        data_uri = [data_uri]
    if not type(data_uri) is list:
        raise Exception('data_uri is neither a string nor a list')
    smiles, Y_names, Y, mask = combine_csv_files(data_uri)
    perm_id = np.random.permutation(len(Y))
    smiles = [smiles[i] for i in perm_id]
    Y = Y[perm_id]
    mask = mask[perm_id]
    
    char_set = _char_set()

    pretrain = not(pretrain_uri==None)
    if pretrain:
        class ArgObj:
            def __init__(self, temp_csv, pretrain_uri):
                self.csv_file = temp_csv
                self.input_vocab_file = ''
                self.output_vocab_file = pretrain_uri
                self.fp_check = False
        with open(temp_csv,'w') as f:
            f.write(','.join( ['SMILES']+Y_names ) + '\n')
            temp = []
            for i in range(len(Y)):
                temp.append( ','.join( [str(j) for j in [smiles[i]]+Y[i].tolist()] ) )
            f.write('\n'.join(temp))
        args = ArgObj(temp_csv, pretrain_uri)
        if not os.path.isfile(pretrain_uri):
            nnfps_main(args)
     
    list_seq = smiles_to_seq(smiles, char_set)
    Xs, X = vectorize(list_seq, char_set)
    
    tstX = X[-ntst:]
    tstXs = Xs[-ntst:]
    tstY = Y[-ntst:]
    tstSmiles = smiles[-ntst:]
    tstMask = mask[-ntst:]

    X = X[:ntrn]
    Xs = Xs[:ntrn]
    Y = Y[:ntrn]
    # we will intentionally randomly corrupt 50% of training data here for code testing purpose
    #Y, mask = _corrupt_training_data(Y) 
    smiles = smiles[:ntrn]
    mask = mask[:ntrn]

    n_trn = int(len(Y)*(1-frac_val))
    n_val = len(Y) - n_trn
    
    trnX = X[:n_trn] 
    trnXs = Xs[:n_trn] 
    trnY = Y[:n_trn] 
    trnSmiles = smiles[:n_trn] 
    trnMask = mask[:n_trn] 

    valX = X[n_trn:n_trn+n_val] 
    valXs = Xs[n_trn:n_trn+n_val] 
    valY = Y[n_trn:n_trn+n_val] 
    valSmiles = smiles[n_trn:n_trn+n_val] 
    valMask = mask[n_trn:n_trn+n_val] 
    
    dim_y = trnY.shape[1]
    trnY_list, valY_list, scalers_Y = [], [], []
    for i in range(dim_y):
        Y_i = np.array([ trnY[j,i] for j in range(len(trnY)) if trnMask[j,i] == 1.0]).reshape(-1,1)
        scaler_Y_i = StandardScaler()
        scaler_Y_i.fit(Y_i)
        trnY_i = scaler_Y_i.transform(trnY[:,i:i+1])
        valY_i = scaler_Y_i.transform(valY[:,i:i+1])
        trnY_list.append(trnY_i)
        valY_list.append(valY_i)
        scalers_Y.append(scaler_Y_i)
    trnY = np.concatenate(trnY_list, axis=1)
    valY = np.concatenate(valY_list, axis=1)

    if pretrain:
        tstXt = smiles_pretrain(tstSmiles, pretrain_uri) 
        Xt = smiles_pretrain(smiles, pretrain_uri) 
        trnXt = Xt[:n_trn] 
        valXt = Xt[n_trn:n_trn+n_val] 

    out = dict()

    out['trnX'] = trnX
    out['trnXs'] = trnXs
    out['trnY'] = trnY
    out['trnSmiles'] = trnSmiles
    out['trnMask'] = trnMask

    out['valX'] = valX
    out['valXs'] = valXs
    out['valY'] = valY
    out['valSmiles'] = valSmiles
    out['valMask'] = valMask

    out['tstX'] = tstX
    out['tstXs'] = tstXs
    out['tstY'] = tstY
    out['tstSmiles'] = tstSmiles
    out['tstMask'] = tstMask

    out['seqlen_x'] = X.shape[1]
    out['dim_x'] = X.shape[2]
    out['dim_y'] = Y.shape[1]

    out['Y_names'] = Y_names

    if pretrain:
        out['trnXt'] = trnXt
        out['valXt'] = valXt
        out['dim_xt'] = Xt.shape[1]

    return out, scalers_Y
    
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

