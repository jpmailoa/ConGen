from __future__ import print_function

import numpy as np
import pandas as pds
import pickle
#from preprocessing import smiles_to_seq, vectorize, smiles_pretrain
from preprocessing import data_preparation, _char_set

from preprocessing import get_property, canonocalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import SSVAE
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# pre-defined parameters
frac=0.5
beta=10000.
char_set = _char_set()
#data_uri=['./data/ZINC_310k.csv', './data/MP_clean_canonize_cut.csv']
data_uri = ['./data/paper_MP_IE_EA.csv','./data/paper_MP_clean_canonize_cut.csv','./data/paper_ZINC_310k.csv','./data/paper_clean_DC.csv','./data/paper_clean_viscosity.csv','./data/paper_pubchem_fluorocarbon.csv']
pretrain_uri='./data/smiles2nn.pkl'
save_uri='./zinc_model.pth.tar'

ntrn=550000
#ntrn=2000
frac_val=0.05
ntst=20000
#ntst=1000


# data preparation
print('::: data preparation')

##data, scaler_Y = data_preparation(data_uri, ntrn, ntst,
##                                  frac_label = frac,
##                                  frac_val = frac_val,
##                                  pretrain_uri=pretrain_uri)
##pickle.dump(data, open('./data/preprocessed_data_bert.pkl','wb'))
##pickle.dump(data, open('./data/preprocessed_scaler_Y_bert.pkl','wb'))
data, scaler_Y = data_preparation(data_uri, ntrn, ntst,
                                  frac_label = frac,
                                  frac_val = frac_val)
#pickle.dump(data, open('./data/preprocessed_data_ori.pkl','wb'))
pickle.dump(scaler_Y, open('./data/preprocessed_scaler_Y_ori.pkl','wb'))
#data = pickle.load(open('./data/preprocessed_data_ori.pkl','rb'))
scaler_Y = pickle.load(open('./data/preprocessed_scaler_Y_ori.pkl','rb'))
#raise Exception('ok quit now')
## model training
print('::: model training')

dim_z = 100
dim_h = 250
n_hidden = 3
batch_size = 60

model = SSVAE.TorchModel(sample_data = data, dim_z = dim_z, dim_h = dim_h,
                         n_hidden = n_hidden, batch_size = batch_size, beta = float(beta), char_set = char_set)
#dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dev = torch.device('cpu')
model.initialize(torch.Tensor(data['trnY']), torch.Tensor(data['trnMask']), dev)
#model.train()

## train the model
#model.train_routine(sample_data = data)
#state = {'state_dict': model.state_dict()}
#torch.save(state, save_uri)

## load trained model
save_uri = 'temp_model_20.pth.tar' 
model.load_state_dict( torch.load(save_uri, map_location=dev)['state_dict'] )

## property prediction performance
scaled_tstY_hat = model.predict_routine(sample_data = data).detach().cpu().numpy()
tstY_hat = [scaler_Y[i].inverse_transform(scaled_tstY_hat[:,i:i+1]) for i in range(scaled_tstY_hat.shape[1])]
tstY_hat = np.concatenate(tstY_hat, axis=1)

dim_y = data['dim_y']
tstY = data['tstY']
tstMask = data['tstMask']
Y_names = data['Y_names']
for j in range(dim_y):
    idx = np.where( tstMask[:,j] == 1 )[0]
    #print([j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])])
    print('Label Name:', Y_names[j])
    print([j, mean_absolute_error(tstY[idx,j], tstY_hat[idx,j])])

## unconditional generation
for t in range(10):
    smi = model.sampling_unconditional()
    print([t, smi, get_property(smi)])

## conditional generation
#ynames = ['n_F', 'MolWt', 'EA']
ynames = ['n_F', 'n_O', 'MolWt', 'IE']
yids = [Y_names.index(yname) for yname in ynames]
#ytargets = [6.0, 250.0, 4.0]
ytargets = [6.0, 2.0, 250.0, 5.5]
ymeans = np.array([scaler_Y[yid].mean_[0] for yid in yids])
ystds = np.array([np.sqrt(scaler_Y[yid].var_[0]) for yid in yids])
ytargets_transform = ( np.array(ytargets) - ymeans ) / ystds
for t in range(10):
    smi = model.sampling_conditional(yids, ytargets_transform)
    print([t, smi, get_property(smi)])
