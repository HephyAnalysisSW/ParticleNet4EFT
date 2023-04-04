data_config_file = 'dataset.yaml'
directory = '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_0.root'
n_split = 1

import numpy as np
import glob
import sys
sys.path.append('..')

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
def get_data(table, indices):
    # inputs
    X = {k: table['_' + k].copy() for k in data_config.input_names}
    # labels
    y = {k: np.array(table[k].copy().tolist(),dtype='float32') for k in data_config.label_names}
    # observers / monitor variables
    Z = {k: np.array(table[k].copy().tolist(),dtype='float32') for k in data_config.z_variables}
    return X, y, Z, indices

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, flist, n=None):
        files = to_filelist(flist)
        n_part = len(files) if n is None else min(n, len(files) )
        self.files = partition( files, n_part )

  def __len__(self):
        return len(self.files)

  def __getitem__(self, index):

        table, indices = _load_next( data_config, self.files[index], (0,1), options={'shuffle':False, 'reweight':False, 'training':True})

        return get_data(table, indices)

dataset = Dataset(directory, n_split)


#load_range_and_fraction ((0, 0.8), 1) file_fraction 1 fetch_by_files True fetch_step 10.0 infinity_mode True in_memory False name train
#
#train_data = SimpleIterDataset(train_file_dict, data_config, for_training=True,
#                               load_range_and_fraction=((0,1.), 1),
#                               file_fraction=1,
#                               fetch_by_files=True,
#                               fetch_step=1,
#                               infinity_mode=True,#args.steps_per_epoch is not None,
#                               in_memory=False,#args.in_memory,
#                               name='train')
#
#from torch.utils.data import DataLoader
#
#train_loader = DataLoader(train_data, batch_size=10**9, drop_last=True, pin_memory=True,
#                          num_workers=1,
#                          persistent_workers=True)
#
#def train_load(flist):
#    """
#    Loads the training data.
#    :param args:
#    :return: train_loader, val_loader, data_config, train_inputs
#    """
#
#    train_file_dict, train_files = to_filelist(args, 'train')
#    if args.data_val:
#        val_file_dict, val_files = to_filelist(args, 'val')
#        train_range = val_range = (0, 1)
#    else:
#        val_file_dict, val_files = train_file_dict, train_files
#        train_range = (0, args.train_val_split)
#        val_range = (args.train_val_split, 1)
#    _logger.info('Using %d files for training, range: %s' % (len(train_files), str(train_range)))
#    _logger.info('Using %d files for validation, range: %s' % (len(val_files), str(val_range)))
#
#    if args.demo:
#        train_files = train_files[:20]
#        val_files = val_files[:20]
#        train_file_dict = {'_': train_files}
#        val_file_dict = {'_': val_files}
#        _logger.info(train_files)
#        _logger.info(val_files)
#        args.data_fraction = 0.1
#        args.fetch_step = 0.002
#
#    if args.in_memory and (args.steps_per_epoch is None or args.steps_per_epoch_val is None):
#        raise RuntimeError('Must set --steps-per-epoch when using --in-memory!')
#
#    train_data = SimpleIterDataset(train_file_dict, args.data_config, for_training=True,
#                                   load_range_and_fraction=(train_range, args.data_fraction),
#                                   file_fraction=args.file_fraction,
#                                   fetch_by_files=args.fetch_by_files,
#                                   fetch_step=args.fetch_step,
#                                   infinity_mode=args.steps_per_epoch is not None,
#                                   in_memory=args.in_memory,
#                                   name='train' + ('' if args.local_rank is None else '_rank%d' % args.local_rank))
#    val_data = SimpleIterDataset(val_file_dict, args.data_config, for_training=True,
#                                 load_range_and_fraction=(val_range, args.data_fraction),
#                                 file_fraction=args.file_fraction,
#                                 fetch_by_files=args.fetch_by_files,
#                                 fetch_step=args.fetch_step,
#                                 infinity_mode=args.steps_per_epoch_val is not None,
#                                 in_memory=args.in_memory,
#                                 name='val' + ('' if args.local_rank is None else '_rank%d' % args.local_rank))
#    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
#                              num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
#                              persistent_workers=args.num_workers > 0 and args.steps_per_epoch is not None)
#    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
#                            num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
#                            persistent_workers=args.num_workers > 0 and args.steps_per_epoch_val is not None)
#    data_config = train_data.config
#    train_input_names = train_data.config.input_names
#    train_label_names = train_data.config.label_names
#
#    return train_loader, val_loader, data_config, train_input_names, train_label_names
#
