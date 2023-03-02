import os
import sys
import glob
import torch
import functools
torch.multiprocessing.set_sharing_strategy('file_system') #https://github.com/pytorch/pytorch/issues/973
from importlib import import_module

# ParticleNet
from utils.logger import _logger, _configLogger
from utils.dataset import SimpleIterDataset
from torch.utils.data import DataLoader

#parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='0',
                    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
parser.add_argument('--samples-per-epoch', type=int, default=None,
                    help='number of samples per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--steps-per-epoch', type=int, default=None,
                    help='number of steps (iterations) per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--log', type=str, default='',
                    help='path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration')
parser.add_argument('--file-fraction', type=float, default=1,
                    help='fraction of files to load; for training, the files are randomly selected for each epoch')
parser.add_argument('-t', '--data-test', nargs='*', default=[],
                    help='testing files; supported syntax:'
                         ' (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;'
                         ' (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;'
                         ' (c) split output per N input files, `--data-test a%10:/path/to/a/*`, will split per 10 input files')
parser.add_argument('-c', '--data-config', type=str, default='data/ak15_points_pf_sv_v0.yaml',
                    help='data config YAML file')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
parser.add_argument('-n', '--network-config', type=str, default='networks/particle_net_pfcand_sv.py',
                    help='network architecture configuration file; the path must be relative to the current dir')
parser.add_argument('-o', '--network-option', nargs=2, action='append', default=[],
                    help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
parser.add_argument('-m', '--model-prefix', type=str, default='models/{auto}/network',
                    help='path to save or load the model; for training, this will be used as a prefix, so model snapshots '
                         'will saved to `{model_prefix}_epoch-%d_state.pt` after each epoch, and the one with the best '
                         'validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path '
                         'including the suffix, otherwise the one with the best validation metric will be used; '
                         'for training, `{auto}` can be used as part of the path to auto-generate a name, '
                         'based on the timestamp and network configuration')
#parser.add_argument('--export-onnx', type=str, default=None,
#                    help='export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); '
#                         'needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)')
parser.add_argument('--load-model-weights', type=str, default=None,
                    help='initialize model with pre-trained weights')
parser.add_argument('--load-epoch', type=int, default=None,
                    help='used to resume interrupted training, load model and optimizer state saved in the `epoch-%d_state.pt` and `epoch-%d_optimizer.pt` files')
parser.add_argument('--tensorboard', type=str, default=None,
                    help='create a tensorboard summary writer with the given comment')
parser.add_argument('--tensorboard-custom-fn', type=str, default=None,
                    help='the path of the python script containing a user-specified function `get_tensorboard_custom_fn`, '
                         'to display custom information per mini-batch or per epoch, during the training, validation or test.')
parser.add_argument('--predict-output', type=str, default='output.root',
                    help='path to save the prediction output, support `.root` and `.awkd` format')

def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ':' in f:
            name, fp = f.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else:
            name, fp = '', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]

    def get_test_loader(name):
        filelist = file_dict[name]
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset({name: filelist}, args.data_config, for_training=False,
                                      load_range_and_fraction=((0, 1), args.data_fraction),
                                      fetch_by_files=True, fetch_step=1,
                                      name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                                 pin_memory=True)
        return test_loader

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config

def flops(model, model_info):
    """
    Count FLOPs and params.
    :param args:
    :param model:
    :param model_info:
    :return:
    """
    from utils.flops_counter import get_model_complexity_info
    import copy

    model = copy.deepcopy(model).cpu()
    model.eval()

    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])

    macs, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=True, verbose=True)
    _logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    _logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

#def onnx(args, model, data_config, model_info):
#    """
#    Saving model as ONNX.
#    :param args:
#    :param model:
#    :param data_config:
#    :param model_info:
#    :return:
#    """
#    assert (args.export_onnx.endswith('.onnx'))
#    model_path = args.model_prefix
#    _logger.info('Exporting model %s to ONNX' % model_path)
#    model = torch.load(args.model_prefix + '_best_epoch_full.pt', map_location=torch.device('cpu'))
#    # model.load_state_dict(torch.load(args.model_prefix + '_best_epoch_full.pt'))#, map_location='cpu'))
#    #model.load_state_dict(torch.load('ctt_lin_quad_test_best_epoch_full.pt'))
#    model = model.cpu()
#    model.eval()
#
#    #os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
#    inputs = tuple(
#        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
#    torch.onnx.export(model, inputs, os.path.join(args.model_directory,args.model_prefix+'_'+args.export_onnx),
#                      input_names=model_info['input_names'],
#                      output_names=model_info['output_names'],
#                      dynamic_axes=model_info.get('dynamic_axes', None),
#                      opset_version=13)
#
#    _logger.info('ONNX model saved to %s', args.export_onnx)
#
#    preprocessing_json = os.path.join(os.path.dirname(args.export_onnx), 'preprocess.json')
#    data_config.export_json(preprocessing_json)
#    _logger.info('Preprocessing parameters saved to %s', preprocessing_json)

def save_root(args, output_path, data_config, scores, labels, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    from utils.data.fileio import _write_root
    output = {}
    if labels[data_config.label_names[0]].ndim>1 and labels[data_config.label_names[0]].shape[1]!=1:
        _logger.warning('Only first column of labels will be written! Vector-valued output not yet supported. Shape: (Nevents, Nvec) = %r', labels[data_config.label_names[0]].shape)
    output[data_config.label_names[0]] = labels[data_config.label_names[0]]

    if scores.ndim>1 and scores.shape[1]!=1:
        _logger.warning('Only first column of scores will be written! Vector-valued output not yet supported. Shape: (Nevents, Nvec) = %r', scores.shape)
    output['output'] = scores
    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v
    for k, v in observers.items():
        if v.ndim > 1:
            # Robert: It is not clear why scalar observers come as [[...]], i.e., with shape (NEvents, 1). 
            #         But we can just flatten.
            if v.ndim==2 and v.shape[1]==1: 
                output[k] = v.flatten()
            else:
                _logger.warning('Ignoring %s, not a 1d array. It has %i dimensions and shape (Nevents, Nvec) = %r', k, v.ndim, v.shape)
                continue
        output[k] = v
    _write_root(output_path, output)

def save_awk(args, output_path, scores, labels, observers):
    """
    Saves as .awkd
    :param scores:
    :param labels:
    :param observers:
    :return:
    """
    from utils.data.tools import awkward
    output = {'scores': scores}
    output.update(labels)
    output.update(observers)

    name_remap = {}
    arraynames = list(output)
    for i in range(len(arraynames)):
        for j in range(i + 1, len(arraynames)):
            if arraynames[i].startswith(arraynames[j]):
                name_remap[arraynames[j]] = '%s_%d' % (arraynames[j], len(name_remap))
            if arraynames[j].startswith(arraynames[i]):
                name_remap[arraynames[i]] = '%s_%d' % (arraynames[i], len(name_remap))
    _logger.info('Renamed the following variables in the output file: %s', str(name_remap))
    output = {name_remap[k] if k in name_remap else k: v for k, v in output.items()}


def model_setup(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config.replace('.py', '').replace('/', '.'))
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    _logger.info('Network options: %s' % str(network_options))
    #if args.export_onnx:
    #    network_options['for_inference'] = True
    #if args.use_amp:
    #    network_options['use_amp'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    if args.load_model_weights:
        model_state = torch.load(args.load_model_weights, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info('Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s' %
                     (args.load_model_weights, missing_keys, unexpected_keys))
    # _logger.info(model)
    flops(model, model_info)
    # loss function
    try:
        loss_func = network_module.get_loss(data_config, **network_options)
        _logger.info('Using loss function %s with options %s' % (loss_func, network_options))
    except AttributeError:
        loss_func = torch.nn.CrossEntropyLoss()
        _logger.warning('Loss function not defined in %s. Will use `torch.nn.CrossEntropyLoss()` by default.',
                        args.network_config)
    return model, model_info, loss_func

if __name__ == '__main__':

    args = parser.parse_args()

    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError('Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!')

    stdout = sys.stdout
    _configLogger('weaver', stdout=stdout, filename=args.log)

#    main(args)
#def main(args):

    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    if args.file_fraction < 1:
        _logger.warning('Use of `file-fraction` is not recommended in general -- prefer using `data-fraction` instead.')

    from utils.nn.tools import evaluate_weighted_regression as evaluate

    # device
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(',')]
        dev = torch.device(gpus[0])
    else:
        gpus = None
        dev = torch.device('cpu')

    print("Using device", dev)

    test_loaders, data_config = test_load(args)

    model, model_info, loss_func = model_setup(args, data_config)

#    # export to ONNX # this was never designed to work with --predict :-)
#    if args.export_onnx:
#        onnx(args, model, data_config, model_info)

    if args.tensorboard:
        from utils.nn.tools import TensorboardHelper
        tb = TensorboardHelper(tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn)
    else:
        tb = None

    orig_model = model
    model = orig_model.to(dev)
    if args.load_epoch is None:
        model_path = args.model_prefix if args.model_prefix.endswith(
            '.pt') else args.model_prefix + '_best_epoch_state.pt'
    else:
        model_path = args.model_prefix if args.model_prefix.endswith(
            '.pt') else args.model_prefix + '_epoch-%d_state.pt'%args.load_epoch
    _logger.info('Loading model %s for eval' % model_path)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    if gpus is not None and len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)
    model = model.to(dev)

    for name, get_test_loader in test_loaders.items():
        test_loader = get_test_loader()
        # run prediction
        if args.model_prefix.endswith('.onnx'):
            _logger.info('Loading model %s for eval' % args.model_prefix)
            from utils.nn.tools import evaluate_onnx
            test_metric, scores, labels, observers = evaluate_onnx(args.model_prefix, test_loader)
        else:
            test_metric, scores, labels, observers = evaluate(
                model, test_loader, dev, epoch=None, for_training=False, tb_helper=tb)
        _logger.info('Test metric %.5f' % test_metric, color='bold')
        del test_loader

        if args.predict_output:
            if '/' not in args.predict_output:
                args.predict_output = os.path.join(
                    os.path.dirname(args.model_prefix),
                    'predict_output', args.predict_output)
            os.makedirs(os.path.dirname(args.predict_output), exist_ok=True)
            if name == '':
                output_path = args.predict_output
            else:
                base, ext = os.path.splitext(args.predict_output)
                output_path = base + '_' + name + ext
            if output_path.endswith('.root'):
                output = save_root(args, output_path, data_config, scores, labels, observers)
            else:
                save_awk(args, output_path, scores, labels, observers)
            _logger.info('Written output to %s' % output_path, color='bold')

