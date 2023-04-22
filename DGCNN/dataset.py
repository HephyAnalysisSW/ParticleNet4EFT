data_config_file = 'dataset.yaml'
directory = '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9].root'
n_split = 5
#n_split = 866//2

import numpy as np
import glob
import sys
sys.path.append('..')
from math import pi

def partition(lst, n):
    ''' Partition list into chunks of approximately equal size'''
    # http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
    n_division = len(lst) / float(n)
    return [ lst[int(round(n_division * i)): int(round(n_division * (i + 1)))] for i in range(n) ]

def to_filelist(flist):
    '''keyword-based: 'a:/path/to/a b:/path/to/b'
    '''
    file_dict = {}
    if type(flist)==type(''):
        flist = [flist]
    for f in flist:
        if ':' in f:
            name, fp = f.split(':')
        else:
            name, fp = '_', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    filelist = sum(file_dict.values(), [])
    assert(len(filelist) == len(set(filelist)))
    return filelist

#from utils.dataset import SimpleIterDataset

from utils.data.config import DataConfig
data_config = DataConfig.load(data_config_file)
from utils.dataset import _load_next

#table, indices = _load_next( data_config, train_files, (0,1),options={'shuffle':False, 'reweight':False, 'training':True})

import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, flist, n_split=None, max_n_files=-1, max_events=None):
        files = to_filelist(flist)
        if max_n_files>0:
            files = files[:max_n_files]
        n_part = len(files) if n_split is None else min(n_split, len(files) )
        self.files = partition( files, n_part )

        self.max_events = max_events

  def get_data(self, table, indices):
        # inputs
        X = {k: table['_' + k][:self.max_events].copy() for k in data_config.input_names}
        # labels
        y = {k: np.array(table[k][:self.max_events].copy().tolist(),dtype='float32') for k in data_config.label_names}
        # observers / monitor variables
        Z = {k: np.array(table[k][:self.max_events].copy().tolist(),dtype='float32') for k in data_config.z_variables}
        return X, y, Z, indices

  def __len__(self):
        return len(self.files)

  def __getitem__(self, index):

        table, indices = _load_next( data_config, self.files[index], (0,1), options={'shuffle':False, 'reweight':False, 'training':True})

        return self.get_data(table, indices)

dataset       = Dataset(directory, n_split)

plot_options =  {
    "parton_hadTop_pt" :{'binning':[50,0,1500], 'tex':'p_{T}(t)'},
    "parton_hadTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t)'},
    "parton_hadTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t)'},
    "parton_lepTop_pt" :{'binning':[30,0,800], 'tex':'p_{T}(t lep)'},
    "parton_lepTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t lep)'},
    "parton_lepTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t lep)'},
    "parton_lepTop_lep_pt" :{'binning':[30,0,800], 'tex':'p_{T}(l (t lep))'},
    "parton_lepTop_lep_eta" :{'binning':[30,-3,3], 'tex':'#eta(l(t lep))'},
    "parton_lepTop_lep_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(l(t lep))'},
    "parton_lepTop_nu_pt" :{'binning':[30,0,800], 'tex':'p_{T}(#nu (t lep))'},
    "parton_lepTop_nu_eta" :{'binning':[30,-3,3], 'tex':'#eta(#nu(t lep))'},
    "parton_lepTop_nu_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(#nu(t lep))'},
    "parton_lepTop_b_pt" :{'binning':[50,0,800], 'tex':'p_{T}(b (t lep))'},
    "parton_lepTop_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t lep))'},
    "parton_lepTop_b_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(b(t lep))'},
    "parton_lepTop_W_pt" :{'binning':[30,0,1000], 'tex':'p_{T}(W (t lep))'},
    "parton_lepTop_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t lep))'},
    "parton_lepTop_W_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(W(t lep))'},

    "delphesJet_pt"                 :{'binning':[50,500,2000], 'tex':'p_{T}(jet)'},
    "delphesJet_eta"                :{'binning':[30,-3,3], 'tex':'#eta(jet)'},
    "delphesJet_phi"                :{'binning':[30,-pi,pi], 'tex':'#phi(jet)'},
    "delphesJet_nConstituents"      :{'binning':[30,30,230], 'tex':'n-constituents'},
    "delphesJet_SDmass"             :{'binning':[30,150,200], 'tex':'M_{SD}(jet)'},
    "delphesJet_SDsubjet0_deltaEta" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaPhi" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaR"   :{'binning':[30,0,0.7], 'tex':'#Delta R(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_mass"     :{'binning':[30,0,200], 'tex':'M_{SD}(jet_{0})'},
    "delphesJet_SDsubjet1_deltaEta" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaPhi" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaR"   :{'binning':[30,0,0.7], 'tex':'#Delta R(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_mass"     :{'binning':[30,0,200], 'tex':'M_{SD}(jet_{1})'},
    "delphesJet_tau1"               :{'binning':[30,0,1], 'tex':'#tau_{1}'},
    "delphesJet_tau2"               :{'binning':[30,0,.5],'tex':'#tau_{2}'},
    "delphesJet_tau3"               :{'binning':[30,0,.3],'tex':'#tau_{3}'},
    "delphesJet_tau4"               :{'binning':[30,0,.3],'tex':'#tau_{4}'},
    "delphesJet_tau21"              :{'binning':[30,0,1], 'tex':'#tau_{21}'},
    "delphesJet_tau32"              :{'binning':[30,0,1], 'tex':'#tau_{32}'},
#https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/RecoJets/JetProducers/python/ECF_cff.py
    "delphesJet_ecf1"               :{'binning':[30,0,2000], 'tex':"ecf1"},
    "delphesJet_ecf2"               :{'binning':[30,0,400000], 'tex':"ecf2"},
    "delphesJet_ecf3"               :{'binning':[30,0,4000000], 'tex':"ecf3"},
    "delphesJet_ecfC1"              :{'binning':[30,0,.5], 'tex':"ecfC1"},
    "delphesJet_ecfC2"              :{'binning':[30,0,.5], 'tex':"ecfC2"},
    "delphesJet_ecfC3"              :{'binning':[30,0,.5], 'tex':"ecfC3"},
    "delphesJet_ecfD"               :{'binning':[30,0,8], 'tex':"ecfD"},
    "delphesJet_ecfDbeta2"          :{'binning':[30,0,20], 'tex':"ecfDbeta2"},
    "delphesJet_ecfM1"              :{'binning':[30,0,0.35], 'tex':"ecfM1"},
    "delphesJet_ecfM2"              :{'binning':[30,0,0.2], 'tex':"ecfM2"},
    "delphesJet_ecfM3"              :{'binning':[30,0,0.2], 'tex':"ecfM3"},
    "delphesJet_ecfM1beta2"         :{'binning':[30,0,0.35], 'tex':"ecfM1beta2"},
    "delphesJet_ecfM2beta2"         :{'binning':[30,0,0.2], 'tex':"ecfM2beta2"},
    "delphesJet_ecfM3beta2"         :{'binning':[30,0,0.2], 'tex':"ecfM3beta2"},
    "delphesJet_ecfN1"              :{'binning':[30,0,0.5], 'tex':"ecfN1"},
    "delphesJet_ecfN2"              :{'binning':[30,0,0.5], 'tex':"ecfN2"},
    "delphesJet_ecfN3"              :{'binning':[30,0,5], 'tex':"ecfN3"},
    "delphesJet_ecfN1beta2"         :{'binning':[30,0,0.5], 'tex':"ecfN1beta2"},
    "delphesJet_ecfN2beta2"         :{'binning':[30,0,0.5], 'tex':"ecfN2beta2"},
    "delphesJet_ecfN3beta2"         :{'binning':[30,0,5], 'tex':"ecfN3beta2"},
    "delphesJet_ecfU1"              :{'binning':[30,0,0.5], 'tex':"ecfU1"},
    "delphesJet_ecfU2"              :{'binning':[30,0,0.04], 'tex':"ecfU2"},
    "delphesJet_ecfU3"              :{'binning':[30,0,0.004], 'tex':"ecfU3"},
    "delphesJet_ecfU1beta2"         :{'binning':[30,0,0.5], 'tex':"ecfU1beta2"},
    "delphesJet_ecfU2beta2"         :{'binning':[30,0,0.04], 'tex':"ecfU2beta2"},
    "delphesJet_ecfU3beta2"         :{'binning':[30,0,0.004], 'tex':"ecfU3beta2"},

    "parton_hadTop_decayAngle_theta" :{'binning':[30,0,pi], 'tex':'#theta(t had)'},
    "parton_hadTop_decayAngle_phi"   :{'binning':[30,-pi,pi], 'tex':'#phi(t had)'},

    "parton_hadTop_q1_pt" :{'binning':[30,0,800], 'tex':'p_{T}(q_{1}(t had))'},
    "parton_hadTop_q1_eta" :{'binning':[30,-3,3], 'tex':'#eta(q_{1}(t had))'},
    "parton_hadTop_q2_pt" :{'binning':[30,0,800], 'tex':'p_{T}(q_{2}(t had))'},
    "parton_hadTop_q2_eta" :{'binning':[30,-3,3], 'tex':'#eta(q_{2}(t had))'},
    "parton_hadTop_b_pt" :{'binning':[30,0,800], 'tex':'p_{T}(b(t had))'},
    "parton_hadTop_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t had))'},
    "parton_hadTop_W_pt" :{'binning':[30,0,800], 'tex':'p_{T}(W(t had))'},
    "parton_hadTop_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t had))'},

    "parton_cosThetaPlus_n"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{n}'},
    "parton_cosThetaMinus_n"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{n}'},
    "parton_cosThetaPlus_r"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{r}'},
    "parton_cosThetaMinus_r"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{r}'},
    "parton_cosThetaPlus_k"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{k}'},
    "parton_cosThetaMinus_k"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{k}'},
    "parton_cosThetaPlus_r_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{n}'},
    "parton_cosThetaMinus_r_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{n}'},
    "parton_cosThetaPlus_k_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{k}'},
    "parton_cosThetaMinus_k_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{k}'},
    "parton_xi_nn"              :{'binning':[30,-1,1], 'tex':'#xi_{nn}'},
    "parton_xi_rr"              :{'binning':[30,-1,1], 'tex':'#xi_{rr}'},
    "parton_xi_kk"              :{'binning':[30,-1,1], 'tex':'#xi_{kk}'},
    "parton_xi_nr_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{+}'},
    "parton_xi_nr_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{-}'},
    "parton_xi_rk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{+}'},
    "parton_xi_rk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{-}'},
    "parton_xi_nk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{+}'},
    "parton_xi_nk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{-}'},
    "parton_cos_phi"            :{'binning':[30,-1,1], 'tex':'cos(#phi)'},
    "parton_cos_phi_lab"        :{'binning':[30,-1,1], 'tex':'cos(#phi lab)'},
    "parton_abs_delta_phi_ll_lab":{'binning':[30,0,pi], 'tex':'|#Delta(#phi(l,l))|'},
}

