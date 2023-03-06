import math
import tqdm
import traceback
from .tools import _concat
from ..logger import _logger
import awkward as ak
import numpy as np
try:
    import uproot3
    #import uproot
except ImportError:
    uproot3 = None
    import uproot
    if uproot.__version__[0] == '3':
        uproot3 = uproot
    else:
        raise ImportError('Please install uproot3 with `pip install uproot3`.')


def _read_hdf5(filepath, branches, load_range=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k:getattr(f.root, k)[:] for k in branches}
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs[branches[0]]))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs


def _read_root(filepath, branches, load_range=None, treename=None):
    with uproot3.open(filepath) as f:
        if treename is None:
            treenames = set([k.decode('utf-8').split(';')[0] for k, v in f.allitems() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 0:
                raise RuntimeError( 'No trees found: %s' % filepath )
            elif len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError('Need to specify `treename` as more than one trees are found in file %s: %s.' % (filepath, str(branches)))
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.numentries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.numentries))
        else:
            start, stop = None, None
        outputs = tree.arrays(branches, namedecode='utf-8', entrystart=start, entrystop=stop)
    return outputs


def _read_awkd(filepath, branches, load_range=None):
    from .tools import awkward
    with awkward.load(filepath) as f:
        outputs = {k: f[k] for k in branches}
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs[branches[0]]))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs


def _read_files(filelist, branches, load_range=None, show_progressbar=False, **kwargs):
    import os
    from collections import defaultdict
    branches = list(branches)
    table = defaultdict(list)
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)
    for filepath in filelist:
        ext = os.path.splitext(filepath)[1]
        if ext not in ('.h5', '.root', '.awkd'):
            raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                a = _read_root(filepath, branches, load_range=load_range, treename=kwargs.get('treename', None))
            elif ext == '.awkd':
                a = _read_awkd(filepath, branches, load_range=load_range)
        except Exception as e:
            a = None
            _logger.error('When reading file %s:', filepath)
            _logger.error(traceback.format_exc())
        if a is not None:
            for name in branches:
                table[name].append(a[name].astype('float32'))
    table = {name:_concat(arrs) for name, arrs in table.items()}
    if len(table[branches[0]]) == 0:
        raise RuntimeError(f'Zero entries loaded when reading files {filelist} with `load_range`={load_range}.')
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    if compression == -1:
        compression = uproot3.write.compress.LZ4(4)
    with uproot3.recreate(file, compression=compression) as fout:
        fout[treename] = uproot3.newtree({k:v.dtype for k, v in table.items()})
        start = 0
        while start < len(list(table.values())[0]) - 1:
            fout[treename].extend({k:v[start:start + step] for k, v in table.items()})
            start += step

def _red( v ):
    # 1D array -> scalar
    if v.ndim == 1:
        return v
    # 2D array, but just one column -> scalar
    elif (v.ndim==2 and v.shape[1]==1):
        return v[:,0]
    # 2D array -> vector
    elif v.ndim==2:
        return v
    else:
        raise RuntimeError("Can not write this shape: %r"%v.shape)

def _write_root_ak( file, table, treename='Events' ):

    if not len(set(list( map( len, table.values())))) == 1:
        raise RuntimeError ("Branches have uneven length!: %r" % list( map( len, table.values())))

    # based on https://awkward-array.org/doc/main/user-guide/how-to-convert-rdataframe.html
    ak.to_rdataframe({k:ak.Array(_red(v)) for k, v in table.items()}).Snapshot( treename, file )

def _write_root4_define(table):
    new_tree = {}
    for k, v in table.items():
        if isinstance(v, np.ndarray):
            t = str(v.dtype)
        else:
            t = f"var * {str(v.type).split()[-1]}"
        new_tree[k.decode("utf-8")] = t
    return new_tree

def _write_root4_convert(table):
    new_tree = {}
    for k, v in table.items():
        if isinstance(v, np.ndarray):
            v1 = v
        else:
            v1 = ak.Array(v)
        new_tree[k.decode("utf-8")] = v1
    return new_tree

def _write_root4(file, table, treename='Events', compression=-1, step=1048576):
    if compression == -1:
        compression = uproot.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        tree = fout.mktree(treename, _write_root4_define(table))
        start = 0
        while start < len(list(table.values())[0]) - 1:
            tree.extend(_write_root4_convert(table))
            start += step
