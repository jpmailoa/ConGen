from __future__ import print_function

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
from transformers import AutoTokenizer, RobertaModel
import pickle

class TorchModel(nn.Module):
    def __init__(self, sample_data, dim_z=100, dim_h=250, n_hidden=3, batch_size=200, beta=10000., char_set=[' '], tmp_model_tag = 'models/default/'):
        super(TorchModel, self).__init__()

        self.seqlen_x = sample_data['seqlen_x']
        self.dim_x = sample_data['dim_x']
        self.dim_y = sample_data['dim_y']
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.beta = beta
        if 'dim_xt' in sample_data:
            self.pretrain = True
            self.dim_xt = sample_data['dim_xt']
        else:
            self.pretrain = False
        
        self.char_to_int = dict((c,i) for i,c in enumerate(char_set))
        self.int_to_char = dict((i,c) for i,c in enumerate(char_set))

        ## sub-models shared between labeled and unlabeled data
        if self.pretrain:
            self.predictor = self._transfer_autofine_predictor()
            self.encoder = self._transfer_encoder()
            self.decoder = self._rnndecoder()       # in: xs,y+z    hid: self.dim_h     out: self.dim_x
        else:
            self.predictor = self._rnnpredictor()     # in: x            hid: self.dim_h     out: 2*self.dim_y
            self.encoder = self._rnnencoder()       # in: x,y          hid: self.dim_h     out: 2*self.dim_z
            self.decoder = self._rnndecoder()       # in: xs,y+z    hid: self.dim_h     out: self.dim_x
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.initialized = False
        self.device = torch.device('cpu')

        self.tmp_model_tag = tmp_model_tag

        return

    def initialize(self, full_y, full_mask, device):
        dim_y = full_y.shape[1]
        self.mu_prior = Parameter( (full_y*full_mask).sum(dim=0) / full_mask.sum(dim=0), requires_grad = False )
        self.cov_prior = Parameter( torch.zeros(dim_y, dim_y), requires_grad = False )
        for i in range(dim_y):
            for j in range(dim_y):
                self.cov_prior[i,j] = torch.sum( (full_y[:,i]-self.mu_prior[i])*full_mask[:,i] * (full_y[:,j]-self.mu_prior[j])*full_mask[:,j] ) / (torch.sum( full_mask[:,i]*full_mask[:,j] )-1)
        self.initialized = True
        self.to(device)
        return
        
    def to(self, device):
        assert self.initialized == True
        super(TorchModel, self).to(device)
        self.device = device
        self.mu_prior = self.mu_prior.to(device)
        self.cov_prior = self.cov_prior.to(device)
        return self

    def print_update(self, log):
        update_file = self.tmp_model_tag+'status.txt'
        if os.path.isfile(update_file):
            with open(update_file, 'a') as f: f.write(log + '\n')
        else:
            with open(update_file, 'w') as f: f.write(log + '\n')
        print(log)
        return

    def forward(self, data):
        assert self.initialized == True
        
        ## functions for incomplete labeled data
        if 'X' in data:               self.x = data['X']
        if 'Xs' in data:             self.xs = data['Xs']
        if 'Y' in data:               self.y = data['Y']
        if 'Xt' in data:              self.xt = data['Xt']
        if 'Smiles' in data:       self.smiles = data['Smiles']
        if 'Mask' in data:         self.mask = data['Mask']
        
        self.predictor_out = self.predictor(data)
        self.y_mu, self.y_lsgms = torch.split(self.predictor_out, [self.dim_y, self.dim_y], dim=-1)
        self.y_sample = self._draw_sample(self.y_mu, self.y_lsgms)
        
        self.y_mu_merged = torch.where( self.mask == 0, self.y_mu, self.y )
        self.y_sample_merged = torch.where( self.mask == 0, self.y_sample, self.y )
        
        self.encoder_out = self.encoder(data, self.y_sample_merged)
        self.z_mu, self.z_lsgms = torch.split(self.encoder_out, [self.dim_z, self.dim_z], dim=-1)
        self.z_sample = self._draw_sample(self.z_mu, self.z_lsgms)
        
        self.decoder_out = self.decoder(data['Xs'], torch.cat([self.z_sample, self.y_sample_merged], dim=-1)) # ??
        self.x_recon = self.softmax(self.decoder_out)

        self.encoder_D_out = self.encoder(data, self.y_mu_merged)
        self.z_D_mu, self.z_D_lsgms = torch.split(self.encoder_D_out, [self.dim_z, self.dim_z], dim=-1)

        self.decoder_D_out = self.decoder(data['Xs'], torch.cat([self.z_D_mu, self.y_mu_merged], dim=-1)) # ??
        self.x_D_recon = self.softmax(self.decoder_D_out)

        cost, cost_val = self.cost_function()
    
        return cost, cost_val

    def cleanup_forward(self):
        del self.predictor_out, self.y_mu, self.y_lsgms, self.y_sample
        del self.y_mu_merged, self.y_sample_merged
        del self.encoder_out, self.z_mu, self.z_lsgms, self.z_sample
        del self.decoder_out, self.x_recon
        del self.encoder_D_out, self.z_D_mu, self.z_D_lsgms
        del self.decoder_D_out, self.x_D_recon
        return

    def cost_function(self):

        # objective functions
        obj = self._obj()
        
        # we average over labeled samples only
        # consider property weight scaling later on
        objYpred_MSE = torch.sum(  ((self.y-self.y_mu)**2) * self.mask ) / torch.sum( self.mask ) 

        obj_val = torch.sum(self.cross_entropy(self.x.flatten(start_dim=1), self.x_D_recon.flatten(start_dim=1)), dim=1).mean()
        
        batch_size = len(self.x)
        
        cost = obj + self.beta * objYpred_MSE
        cost_val = objYpred_MSE
        
        self.print_update('--Cost Train LU ' + str(float(obj.detach().cpu().numpy())))
        self.print_update('--Cost Train Y ' + str(float(objYpred_MSE.detach().cpu().numpy() * self.beta)))
        
        return cost, cost_val
            
    def train_routine(self, sample_data):
        assert self.initialized == True

        trnX = sample_data['trnX']
        valX = sample_data['valX']
        
        # batch sizes
        batch_size = int(self.batch_size)
        n_batch = math.ceil(len(trnX)/batch_size)
        val_split = math.ceil(len(valX)/batch_size)
        batch_size_val = math.ceil(len(valX)/val_split)

        # optimizer
        if self.pretrain:
            optimizer = optim.Adam( [{'params': self.encoder.parameters()},
                                     {'params': self.decoder.parameters()},
                                     {'params': self.predictor.parameters(), 'lr': 3e-5}],
                                    lr=1e-3, weight_decay=0 )
        else:
            optimizer = optim.Adam( self.parameters(), 1e-4, weight_decay=0 )

        update_file = self.tmp_model_tag+'status.txt'
        if os.path.isfile(update_file):
            os.system('rm '+update_file)
            
        # training
        n_iter = 300
        val_log=np.zeros(n_iter)
        
        for epoch in range(n_iter):
            self.print_update('Step: '+str(epoch)+'/'+str(n_iter))

            ## For debugging
##            for textword in locals():
##                self.print_update('Textword: '+str(textword))
            permid_trn, permid_val = self._forward_permutation_idx(sample_data)
            self.train()

            trn_temp_y = []

            self.print_update('Training Cycle')
            for i in range(n_batch):
                self.print_update('Batch: '+str(i)+'/'+str(n_batch))
                
                start = i*batch_size
                end = min(start + batch_size, len(trnX))
                batch_permid_trn = permid_trn[start:end]
                
                train_batch = self._tensor_indexed_data_lowmem( sample_data, 'trn', batch_permid_trn )
                cost, cost_val = self.forward( train_batch )

                # store self.y_mu_merged to compute new self.mu_prior and self.cov_prior
                trn_temp_y.append( self.y_mu_merged.detach().cpu().numpy() )
                
                # compute gradient and backward propagation
                optimizer.zero_grad()
                cost.backward()
                if not self.pretrain:
                    nn.utils.clip_grad_norm_( self.parameters(), 1e2 )
                
                # debugging
                nan_flag = False
                for k,v in self.named_parameters():
                    if not( v.grad is None ):
                        if torch.isnan(v.grad).any():
                            self.print_update( str(k) + ' has nan!!!')
                            nan_flag = True
                    del k,v
##                if nan_flag:
##                    for k in range(len(self.cross_temp)):
##                        self.print_update(str(k))
##                        if self.cross_temp[k].grad == None:
##                            self.print_update('None')
##                            continue
##                        if torch.isnan(self.cross_temp[k].grad).any():
##                            self.print_update('grad of cross_temp '+str(k)+' has nan!!!!')
##                            temp = self.cross_temp[k].grad.detach().cpu().numpy()
##                            pickle.dump(temp, open('grad_'+str(k)+'.pkl','wb'))
##                            del temp
##                del self.cross_temp

                if nan_flag:
                    self.print_update('Skipping epoch '+str(epoch)+' batch '+str(i)+' grad update because of nan issue')
                else:
                    optimizer.step()
                
                trn_res = float(cost.detach().cpu().numpy())

                del train_batch, cost, cost_val
                self.cleanup_forward()

            self.eval()
            
            val_res = []

            self.print_update('Validation Cycle')
            for i in range(val_split):
                self.print_update('Batch: '+str(i)+'/'+str(val_split))
                                  
                start = i*batch_size_val
                end = min(start + batch_size_val, len(valX))
                batch_permid_val = permid_val[start:end]

                val_batch = self._tensor_indexed_data_lowmem( sample_data, 'val', batch_permid_val )

                with torch.no_grad():
                    cost, cost_val = self.forward( val_batch )
                    val_res.append( cost_val.detach().cpu().numpy() )

                del val_batch, cost, cost_val
                self.cleanup_forward()
            
            
            val_res = np.mean(val_res,axis=0)
            self.print_update(str(epoch) + '[\'Training\', \'cost_trn\', ' + str(trn_res) + ']')
            self.print_update('--- [\'Validation\', \'cost_val\', ' + str(val_res) + ']')
            
            val_log[epoch] = val_res
            if epoch > 20 and np.min(val_log[0:epoch-10]) * 0.99 < np.min(val_log[epoch-10:epoch+1]):
                self.print_update('---termination condition is met')
                break

            trn_temp_y = torch.Tensor( np.concatenate( trn_temp_y, axis=0 ) )
            trn_mask_y = torch.ones( trn_temp_y.shape )
            self.initialize(trn_temp_y, trn_mask_y, self.device)
            del trn_temp_y, trn_mask_y

            state = {'state_dict': self.state_dict()}
            torch.save(state, self.tmp_model_tag+'model_'+str(epoch)+'.pth.tar')
            
        return

    def predict_routine(self, sample_data):
        assert self.initialized == True

        self.eval()
        n_iter = math.ceil(sample_data['tstX'].shape[0] / self.batch_size)
        y_mu = []

        for i in range(n_iter):
            start = i * self.batch_size
            end = min(start + self.batch_size, sample_data['tstX'].shape[0])
            tst_batch = self._tensor_predict_indexed_data(sample_data, start, end)
                
            self.predictor_out = self.predictor(tst_batch)
            self.y_mu, self.y_lsgms = torch.split(self.predictor_out, [self.dim_y, self.dim_y], dim=-1)

            y_mu.append( self.y_mu.detach() )

        y_mu = torch.cat(y_mu, dim=0)
            
        return y_mu

    def latent(self, data_input, y_input):
        assert self.initialized == True

        self.y = torch.Tensor(y_input).to(self.device)
        
        self.predictor_out = self.predictor(data_input)
        self.y_mu, self.y_lsgms = torch.split(self.predictor_out, [self.dim_y, self.dim_y], dim=-1)
        
        self.encoder_out = self.encoder(data_input, self.y)
        self.z_mu, self.z_lsgms = torch.split(self.encoder_out, [self.dim_z, self.dim_z], dim=-1)

        return self.z_mu

    def sampling_unconditional(self): 
       
        sample_z = np.random.randn(1, self.dim_z)
        mu_prior = self.mu_prior.detach().cpu().numpy()
        cov_prior = self.cov_prior.detach().cpu().numpy()
        sample_y = np.random.multivariate_normal(mu_prior, cov_prior, 1)      
          
        sample_smiles = self.beam_search(sample_z, sample_y, k=5)

        return sample_smiles

    def sampling_conditional(self, yids, ytargets):
        # both yid and ytarget are list of integers
        for yid in yids:    assert  ( (type(yid) is int) and (yid in range(self.dim_y)) )
        assert len(yids) == len(ytargets)
        assert len(np.unique(np.array(yids).astype(int))) == len(yids)  # ensures each yid is unique

        if len(yids) == 0:
            return self.sampling_unconditional()

        mu_prior = self.mu_prior.detach().cpu().numpy()
        cov_prior = self.cov_prior.detach().cpu().numpy()
        
        def random_cond_normal(yids, ytargets):

            # if all properties are designated as conditions, do not do sampling
            if len(yids) == self.dim_y:
                tst = np.zeros(self.dim_y)
                tst[yids] = ytargets
                return np.asarray([tst])
            
            id2 = yids
            id1 = [yid for yid in range(self.dim_y) if (not yid in yids)]

            mu1 = mu_prior[id1]
            mu2 = mu_prior[id2]
            
            cov11 = cov_prior[id1][:,id1]
            cov12 = cov_prior[id1][:,id2]
            cov22 = cov_prior[id2][:,id2]
            cov21 = cov_prior[id2][:,id1]
            
            cond_mu = np.transpose(mu1.reshape([1,-1]).T+np.matmul(cov12, np.linalg.inv(cov22)) * (ytargets-mu2))[0]
            cond_cov = cov11 - np.matmul(np.matmul(cov12, np.linalg.inv(cov22)), cov21)
            
            marginal_sampled = np.random.multivariate_normal(cond_mu, cond_cov, 1)
            
            tst = np.zeros(self.dim_y)
            tst[id1] = marginal_sampled
            tst[id2] = ytargets
            
            return np.asarray([tst])

        sample_z = np.random.randn(1, self.dim_z)
        sample_y = random_cond_normal(yids, ytargets) 
          
        sample_smiles = self.beam_search(sample_z, sample_y, k=5)
            
        return sample_smiles

    def beam_search(self, z_input, y_input, k=5):

        def reconstruct(xs_input, z_sample, y_input):

            assert self.initialized == True

            self.xs, self.z_G, self.y = xs_input, z_sample, y_input
            
            self.decoder_G_out = self.decoder(self.xs, torch.cat([self.z_G, self.y], dim=-1))
            self.x_G_recon = self.softmax(self.decoder_G_out)

            return self.x_G_recon
        
        cands = np.asarray([np.zeros((1, self.seqlen_x, self.dim_x), dtype=np.float32)] )
        cands_score = np.asarray([100.])
        
        for i in range(self.seqlen_x-1):
        
            cands2 = []
            cands2_score = []

            for j, samplevec in enumerate(cands):
                o = reconstruct(torch.Tensor(samplevec).to(self.device),
                                torch.Tensor(z_input).to(self.device),
                                torch.Tensor(y_input).to(self.device))

                o = o.detach().cpu().numpy()
                
                sampleidxs = np.argsort(-o[0,i])[:k]
                
                for sampleidx in sampleidxs: 
                    
                    samplevectt = np.copy(samplevec)
                    samplevectt[0, i+1, sampleidx] = 1.
                    
                    cands2.append(samplevectt)
                    cands2_score.append(cands_score[j] * o[0,i,sampleidx])
                    
            cands2_score = np.asarray(cands2_score)
            cands2 = np.asarray(cands2)
            
            kbestid = np.argsort(-cands2_score)[:k]
            cands = np.copy(cands2[kbestid])
            cands_score = np.copy(cands2_score[kbestid])
            
            if np.sum([np.argmax(c[0][i+1]) for c in cands]) == 0:
                break

        sampletxt = ''.join([self.int_to_char[np.argmax(t)] for t in cands[0,0]]).strip()

        return sampletxt

    def _obj_L(self):

        L_log_lik = - torch.sum(self.cross_entropy( self.x_L.flatten(start_dim=1), self.x_L_recon.flatten(start_dim=1)), dim=1)
        L_log_prior_y = self.noniso_logpdf(self.y_L)
        L_KLD_z = self.iso_KLD(self.z_L_mu, self.z_L_lsgms)

        objL = - torch.mean(L_log_lik + L_log_prior_y - L_KLD_z)
        
        return objL


    def _obj_U(self):

        U_log_lik = - torch.sum(self.cross_entropy( self.x_U.flatten(start_dim=1), self.x_U_recon.flatten(start_dim=1)), dim=1)
        U_KLD_y = self.noniso_KLD(self.y_U_mu, self.y_U_lsgms)
        U_KLD_z = self.iso_KLD(self.z_U_mu, self.z_U_lsgms)

        objU = - torch.mean(U_log_lik - U_KLD_y - U_KLD_z)
        
        return objU

    def _obj(self):

        log_lik = - torch.sum(self.cross_entropy( self.x.flatten(start_dim=1), self.x_recon.flatten(start_dim=1)), dim=1)
        log_prior_y = self.noniso_logpdf(self.y, self.mask)
        KLD_y = self.noniso_KLD(self.y_mu, self.y_lsgms, self.mask)
        KLD_z = self.iso_KLD(self.z_mu, self.z_lsgms)

        obj = -torch.mean(log_lik + log_prior_y - KLD_y - KLD_z)
##        self.print_update(str(torch.mean(log_lik)))
##        self.print_update(str(torch.mean(log_prior_y)))
##        self.print_update(str(torch.mean(KLD_y)))
##        self.print_update(str(torch.mean(KLD_z)))
##        self.print_update(str(self.cov_prior.inverse() ))
        return obj

    def cross_entropy(self, x, y, const = 1e-10):
        return - ( x*y.clip(const, 1.0).log() + (1.0-x)*((1.0-y).clip(const, 1.0).log()) )
##        t1 = x
##        t2 = y
##        t3 = t2.clip(const, 1.0)
##        t4 = t3.log()
##        t5 = t1*t4
##        t6 = 1.0-t1
##        t7 = 1.0-t2
##        t8 = t7.clip(const, 1.0)
##        t9 = t8.log()
##        t10 = t6*t9
##        t11 = -(t5 + t10)
##        self.print_update('t2 grad: '+str(t2.requires_grad))
##        if t2.requires_grad:
##            t2.retain_grad()
##            t3.retain_grad()
##            t4.retain_grad()
##            t5.retain_grad()
##            t7.retain_grad()
##            t8.retain_grad()
##            t9.retain_grad()
##            t10.retain_grad()
##            t11.retain_grad()
##        self.cross_temp = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11]
##        return t11
        
        
    def iso_KLD(self, mu, log_sigma_sq):
        return torch.sum( -0.5 * (1.0 + log_sigma_sq - mu**2 - log_sigma_sq.exp()), dim=1)
        
    def noniso_logpdf(self, x, mask):
        # pick only the labeled components
        diff = torch.where( mask==0, torch.zeros(mask.shape).to(self.device), x - self.mu_prior )
        return - 0.5 * (float(self.cov_prior.shape[0]) * np.log(2.*np.pi) #  +  self.cov_prior.det().log()
                        + torch.sum( torch.matmul( diff, self.cov_prior.inverse() ) * diff, dim=1))
        
    def noniso_KLD(self, mu, log_sigma_sq, mask):
        # pick only the unlabeled components
        diff = torch.where( mask==0, self.mu_prior - mu, torch.zeros(mask.shape).to(self.device) )
        inv = self.cov_prior.inverse()
        masked_log_sigma_sq = torch.where( mask==0, log_sigma_sq, torch.zeros(mask.shape).to(self.device) )
        masked_log_sigma_sq_exp = torch.where( mask==0, log_sigma_sq.exp(), torch.zeros(mask.shape).to(self.device) )
        return 0.5 * ( torch.stack([torch.matmul(inv,x).trace() for x in masked_log_sigma_sq_exp.diag_embed()])  
                      + torch.sum( torch.matmul(diff, self.cov_prior.inverse()) * diff, dim=1 )  
                      # - float(self.cov_prior.shape[0]) + self.cov_prior.det().log() - torch.sum(masked_log_sigma_sq, dim=1) )
                      - float(self.cov_prior.shape[0]) - torch.sum(masked_log_sigma_sq, dim=1) )

    def _permutation(self, set):

        permid=np.random.permutation(len(set[0]))
        for i in range(len(set)):
            if type(set[i]) is list:
                set[i] = [set[i][j] for j in permid]
            else:
                set[i] = set[i][permid]

        return set

    def _forward_permutation_idx(self, dict_obj):
        assert type(dict_obj) == dict
        keys = [key for key in dict_obj]
        permid_trn, permid_val = None, None
        temp_dict = dict()
        for i in range(len(keys)):
            key = keys[i]
            if key.startswith('trn'):
                if permid_trn is None:
                    permid_trn = np.random.permutation(len(dict_obj[key]))
            elif key.startswith('val'):
                if permid_val is None:
                    permid_val = np.random.permutation(len(dict_obj[key]))
        return permid_trn, permid_val

    def _tensor_indexed_data_lowmem(self, dict_obj, mode, permid):
        assert type(dict_obj) == dict
        assert mode in ['trn', 'val']
        keys = [key for key in dict_obj]
        out_dict = dict()
        for i in range(len(keys)):
            key = keys[i]
            if key.startswith(mode):
                if type(dict_obj[key]) is list:
                    temp = [dict_obj[key][j] for j in permid]
                else:
                    temp = torch.Tensor( dict_obj[key][permid] ).to(self.device)
                out_dict[ key[3:] ] = temp
        return out_dict 

    def _forward_permutation(self, dict_obj):
        assert type(dict_obj) == dict
        keys = [key for key in dict_obj]
        permid_trn, permid_val = None, None
        temp_dict = dict()
        for i in range(len(keys)):
            key = keys[i]
            if key.startswith('trn'):
                if permid_trn is None:
                    permid_trn = np.random.permutation(len(dict_obj[key]))
                permid = permid_trn
            elif key.startswith('val'):
                if permid_val is None:
                    permid_val = np.random.permutation(len(dict_obj[key]))
                permid = permid_val
            else:
                # don't modify content sequence, just leave
                temp_dict[key] = dict_obj[key]
                continue
            if type(dict_obj[key]) is list:
                temp_dict[key] = [dict_obj[key][j] for j in permid]
            else:
                temp_dict[key] = dict_obj[key][permid]
        out_dict = dict()
        out_dict['trn'] = dict()
        out_dict['val'] = dict()
        for key in keys:
            if key.startswith('trn'):
                out_dict['trn'][ key[3:] ] = temp_dict[key]
            elif key.startswith('val'):
                out_dict['val'][ key[3:] ] = temp_dict[key]
        return out_dict        

    def _tensor_indexed_data(self, data, start, end):
        out = dict()
        for key in data.keys():
            if type(data[key]) is list:
                out[key] = data[key][start:end]
            else:
                out[key] = torch.Tensor( data[key][start:end] ).to(self.device)
        return out

    def _tensor_predict_indexed_data(self, data, start, end):
        out = {}
        for key in data.keys():
            if key.startswith('tst'):
                temp = data[key][start:end]
                if not (type(temp) is list):
                    temp = torch.Tensor(temp).to(self.device)
                out[ key[3:] ] = temp
        return out

    def _draw_sample(self, mu, lsgms):

        epsilon = torch.normal(mean=torch.zeros(mu.shape).to(self.device), std=torch.ones(mu.shape).to(self.device))
        sample = mu + epsilon*torch.exp(0.5*lsgms)

        return sample 


    def _rnnpredictor(self):

        class bi_GRU_zero(nn.Module):
            def __init__(subs):
                super(bi_GRU_zero, subs).__init__()
                subs.GRU = nn.GRU(input_size = self.dim_x,
                                  hidden_size = self.dim_h,
                                  num_layers = self.n_hidden,
                                  bias = True,
                                  batch_first = True,
                                  dropout = 0,
                                  bidirectional = True)
                subs.linear_out_y = nn.Linear(in_features = 2*self.dim_h, out_features = 2*self.dim_y)
                return
                                        
            def forward(subs, data):
                x = data['X']
                output, hidden = subs.GRU(x)
                hidden = hidden.view(hidden.shape[0]//2, 2, hidden.shape[1], hidden.shape[2])
                hidden = hidden[-1]
                hidden_forward, hidden_backward = hidden[0], hidden[1]
                fc_input = torch.cat((hidden_forward, hidden_backward), dim=1)
                out = subs.linear_out_y( fc_input )
                return out

        return bi_GRU_zero()


    def _rnnencoder(self):

        class bi_GRU_init(nn.Module):
            def __init__(subs):
                super(bi_GRU_init, subs).__init__()
                subs.GRU = nn.GRU(input_size = 2*self.dim_x,
                                  hidden_size = self.dim_h,
                                  num_layers = self.n_hidden,
                                  bias = True,
                                  batch_first = True,
                                  dropout = 0,
                                  bidirectional = True)
                subs.linear_in = nn.Linear(in_features = self.dim_y, out_features = 2*self.dim_h)
                subs.linear_peek = nn.Linear(in_features = self.dim_y, out_features = self.dim_x)
                subs.linear_out_z = nn.Linear(in_features = 2*self.dim_h, out_features = 2*self.dim_z)
                return

            def forward(subs, data, st):
                x = data['X']
                h_0 = self.sigmoid( subs.linear_in(st) ).reshape(st.shape[0], 2, self.dim_h).transpose(0,1)
                h_0 = h_0.repeat(self.n_hidden, 1, 1)
                peek_in = self.sigmoid( subs.linear_peek(st) )
                peek = peek_in.tile(1, self.seqlen_x).reshape(-1, self.seqlen_x, self.dim_x)
                output, hidden = subs.GRU(torch.cat([x,peek],dim=2), h_0)
                hidden = hidden.view(hidden.shape[0]//2, 2, hidden.shape[1], hidden.shape[2])
                hidden = hidden[-1]
                hidden_forward, hidden_backward = hidden[0], hidden[1]
                fc_input = torch.cat((hidden_forward, hidden_backward), dim=1)
                out = subs.linear_out_z( fc_input )
                return out
                
        return bi_GRU_init()

    def _rnndecoder(self): #, x, st, dim_h, dim_y, reuse=False):

        class uni_GRU_init(nn.Module):
            def __init__(subs):
                super(uni_GRU_init, subs).__init__()
                subs.GRU = nn.GRU(input_size = 2*self.dim_x,
                                  hidden_size = self.dim_h,
                                  num_layers = self.n_hidden,
                                  bias = True,
                                  batch_first = True,
                                  dropout = 0,
                                  bidirectional = False)
                subs.linear_in = nn.Linear(in_features = self.dim_y+self.dim_z, out_features = self.dim_h)
                subs.linear_peek = nn.Linear(in_features = self.dim_y+self.dim_z, out_features = self.dim_x)
                subs.linear_out_x = nn.Linear(in_features = self.dim_h, out_features = self.dim_x)
                return

            def forward(subs, x, st):
                h_0 = self.sigmoid( subs.linear_in(st) ).reshape(st.shape[0], 1, self.dim_h).transpose(0,1)
                h_0 = h_0.repeat(self.n_hidden, 1, 1)
                peek_in = self.sigmoid( subs.linear_peek(st) )
                peek = peek_in.tile(1, self.seqlen_x).reshape(-1, self.seqlen_x, self.dim_x)
                output, hidden = subs.GRU(torch.cat([x,peek],dim=2), h_0)
                out = subs.linear_out_x( output )
                return out

        return uni_GRU_init()


    def _transfer_predictor(self):

        class transfer_linear_zero(nn.Module):
            def __init__(subs):
                super(transfer_linear_zero, subs).__init__()
                # should use transfer learning representation
                subs.linear_in_x = nn.Linear(in_features = self.dim_xt, out_features = self.dim_h)
                subs.linear_h = nn.Linear(in_features = self.dim_h, out_features = self.dim_h)
                subs.relu = nn.ReLU()
                subs.linear_out_y = nn.Linear(in_features = self.dim_h, out_features = 2*self.dim_y)
                return
                                        
            def forward(subs, data):
                Xt = data['Xt']
                out = subs.linear_in_x(Xt)
                out = subs.relu(out)
                out = subs.linear_h(out)
                out = subs.relu(out)
                out = subs.linear_out_y(out)
                return out

        return transfer_linear_zero()

    def _transfer_encoder(self):

        class transfer_linear_init(nn.Module):
            def __init__(subs):
                super(transfer_linear_init, subs).__init__()
                # should use transfer learning representation
                subs.linear_in_x = nn.Linear(in_features = self.dim_xt, out_features = self.dim_h)
                subs.linear_in_y = nn.Linear(in_features = self.dim_y, out_features = self.dim_h)
                subs.linear_h = nn.Linear(in_features = 2*self.dim_h, out_features = self.dim_h)
                subs.relu = nn.ReLU()
                subs.linear_out_z = nn.Linear(in_features = self.dim_h, out_features = 2*self.dim_z)
                return
                                        
            def forward(subs, data, Y):
                Xt = data['Xt']
                in1 = subs.linear_in_x(Xt)
                in1 = subs.relu(in1)
                in2 = subs.linear_in_y(Y)
                in2 = subs.relu(in2)
                out = torch.cat([in1, in2], dim=1)
                out = subs.linear_h(out)
                out = subs.relu(out)
                out = subs.linear_out_z(out)
                return out

        return transfer_linear_init()

    def _transfer_autofine_predictor(self):

        class transfer_linear_zero(nn.Module):
            def __init__(subs):
                super(transfer_linear_zero, subs).__init__()
                # should use transfer learning representation
                subs.model = RobertaModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
                subs.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
                subs.linear_in_x = nn.Linear(in_features = self.dim_xt, out_features = self.dim_h)
                subs.relu = nn.ReLU()
                subs.linear_out_y = nn.Linear(in_features = self.dim_h, out_features = 2*self.dim_y)
                #subs.freeze_layers( [-1,0,1,2,3,4,5] )
                subs.freeze_layers( [-1,0,1,2,3,4] )
                return
                                        
            def forward(subs, data):
                smiles_list = data['Smiles']
                Xt = []
                for smiles in smiles_list:
                    inputs = subs.tokenizer.encode_plus(smiles, return_tensors='pt', add_special_tokens=True)
                    input_ids = inputs['input_ids'].to(self.device)
                    nn_fp = subs.model(input_ids)['pooler_output']
                    Xt.append(nn_fp)
                Xt = torch.cat(Xt, dim=0)
                out = subs.linear_in_x(Xt)
                out = subs.relu(out)
                out = subs.linear_out_y(out)
                return out

            def freeze_layers(subs, freeze_indices):
                for i in freeze_indices:
                    if i < 0:
                        params = subs.model.embeddings.parameters()
                    elif i in range(6):
                        params = subs.model.encoder.layer[i].parameters()
                    else:
                        raise Exception('Invalid freeze indices')
                    for p in params: p.requires_grad = False
                return

        return transfer_linear_zero()
