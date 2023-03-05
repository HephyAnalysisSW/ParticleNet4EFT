import yaml
import copy
import glob
from ..logger import _logger

def _as_list(x):
    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


def _md5(fname):
    '''https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file'''
    import hashlib
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class PredictConfig(object):
    r"""Class to hold predict configuration.
    """

    def __init__(self, print_info=True, **kwargs):

        self.models = kwargs
        if print_info:
            _logger.debug(self.models)

        self.shortnames = {}

        for name, model_conf in self.models.items():
#            if 'network-options' not in model_conf:
#                model_conf['network-options'] = []
            if not name in self.shortnames:
                self.shortnames[name] = {}
            else:
                raise RuntimeError("Duplicate entry in predict-config: %s" % name)

            if 'epochs' not in model_conf:
                model_conf['epochs'] = ['best']
            files = glob.glob(model_conf['prefix'] + '_epoch-*_state.pt')
            if 'all' in model_conf['epochs']:
                model_conf['states'] = files + [model_conf['prefix'] + '_best_epoch_state.pt']
            else:
                model_conf['states'] = []
                if -1 in model_conf['epochs']:
                    model_conf['states'].append(max( files, key = lambda f: int(f.split('-')[-1].split('_')[0]) if 'epoch-' in f else -1 ))
                if "best" in model_conf['epochs']:
                    model_conf['states'].append( model_conf['prefix'] + '_best_epoch_state.pt' )
            
                for file_ in files:
                    if file_.split('-')[-1].split('_')[0] in map( str, model_conf['epochs']) and file_ not in model_conf['states']:
                        model_conf['states'].append(file_)

            for f_ in model_conf['states']:
                if '_best_epoch' in f_:
                    self.shortnames[(name, f_)] = name.replace('-','_')+'_best'
                else:
                    self.shortnames[(name, f_)] = name.replace('-','_')+'_epoch_%i'% int(f_.split('-')[-1].split('_')[0])

    def __getattr__(self, name):
        return self.models[name]

#    def dump(self, fp):
#        with open(fp, 'w') as f:
#            yaml.safe_dump(self.options, f, sort_keys=False)

    @classmethod
    def load(cls, fp, load_observers=True):
        with open(fp) as f:
            models = yaml.safe_load(f)
        return cls(**models)

    def copy(self):
        return self.__class__(print_info=False, **copy.deepcopy(self.models))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

#    def export_json(self, fp):
#        import json
#        j = {'output_names':self.label_value, 'input_names':self.input_names}
#        for k, v in self.input_dicts.items():
#            j[k] = {'var_names':v, 'var_infos':{}}
#            for var_name in v:
#                j[k]['var_length'] = self.preprocess_params[var_name]['length']
#                info = self.preprocess_params[var_name]
#                j[k]['var_infos'][var_name] = {
#                    'median': 0 if info['center'] is None else info['center'],
#                    'norm_factor': info['scale'],
#                    'replace_inf_value': 0,
#                    'lower_bound': -1e32 if info['center'] is None else info['min'],
#                    'upper_bound': 1e32 if info['center'] is None else info['max'],
#                    'pad': info['pad_value']
#                    }
#        with open(fp, 'w') as f:
#            json.dump(j, f, indent=2)
