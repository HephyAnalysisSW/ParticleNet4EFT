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
                    help='testing files; supported syntax  plain list, `--data-test /path/to/a/* /path/to/b/*`;')
parser.add_argument('-c', '--data-config', type=str, default='data/ak15_points_pf_sv_v0.yaml',
                    help='data config YAML file')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
parser.add_argument('-p', '--predict-config', type=str, default='HadronicSMEFT/predict/predict.yaml',
                    help='predict config YAML file')
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
parser.add_argument('--nJobs', action='store', nargs='?', type=int, default=1,  help="Maximum number of simultaneous jobs.")
parser.add_argument('--job',   action='store', nargs='?', type=int, default=0,  help="Run only job i")

def partition(lst, n):
    ''' Partition list into chunks of approximately equal size'''
    # http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
    n_division = len(lst) / float(n)
    return [ lst[int(round(n_division * i)): int(round(n_division * (i + 1)))] for i in range(n) ]

def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """

    files = []
    for f in args.data_test:
        files += glob.glob(f)
    files = sorted(files)

    if args.nJobs>1:
        n_files_before = len(files) 
        files = partition( files, args.nJobs )[args.job] 
        n_files_after  = len(files)
        _logger.info( "Running job %i/%i over %i files from a total of %i.", args.job, args.nJobs, n_files_after, n_files_before)
    else:
        _logger.info( "Running over all %i files." % len(files) )

    file_dict = {'': files }

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

    macs, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=False, verbose=False)
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

def save_root(args, output_path, scores, labels, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    from utils.data.fileio import _write_root_ak
    output = scores
    output.update( labels )
    output.update( observers )
    output = {k.replace('-','_'):v for k,v in output.items()}
    _write_root_ak(output_path, output)

if __name__ == '__main__':

    args = parser.parse_args()

    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError('Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!')

    stdout = sys.stdout
    _configLogger('weaver', stdout=stdout, filename=args.log)
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

    if args.tensorboard:
        from utils.nn.tools import TensorboardHelper
        tb = TensorboardHelper(tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn)
    else:
        tb = None

    from utils.predict.config import PredictConfig
    predict_config = PredictConfig.load(args.predict_config)

    models = {}
    for model_conf in predict_config.models.values():
        network_module = import_module(model_conf['network-config'].replace('.py', '').replace('/', '.'))
        model_conf['models'] = {}
        for state in model_conf['states']:

            model, model_info = network_module.get_model(data_config)#, **network_options)
            model_state = torch.load(state, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            _logger.info('Model %s initialized with weights from %s' % (model_conf['network-config'], state))
            if missing_keys:
                _logger.warning( "Missing keys: %s", missing_keys)
            if unexpected_keys:
                _logger.warning( "Unexpected keys: %s", unexpected_keys)
            # _logger.info(model)
            flops(model, model_info)
            model = model.to(dev)
            model_conf['models'][state] = {'model':model, 'model_info':model_info, 'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys} 
                
    for name, get_test_loader in test_loaders.items():
        test_loader = get_test_loader()

        scores, labels = {}, {}
        for model_name, model_conf in predict_config.models.items():
            for state in model_conf['states']:
                key = predict_config.shortnames[(model_name, state)] 
                _, scores[key], labels[model_name], observers = evaluate(
                    model_conf['models'][state]['model'], test_loader, dev, epoch=None, for_training=False, tb_helper=tb)
            labels[model_name]

        del test_loader

        if len(list(filter( lambda l:l!=1, list(map( len, labels.values()))))):
            raise RuntimeError("Not all labels are unique. ")
        labels = {k:list(v.values())[0] for k,v in labels.items()}

        if '/' not in args.predict_output:
            args.predict_output = os.path.join(
                #os.path.dirname(args.model_prefix),
                'predict_output', args.predict_output)
        os.makedirs(os.path.dirname(args.predict_output), exist_ok=True)
        base, ext = os.path.splitext(args.predict_output)
        if args.nJobs == 1:
            output_path = base + ext
        else:
            output_path = base + '_' + str(args.job) + ext
        if output_path.endswith('.root'):
            save_root(args, output_path, scores, labels, observers)
        else:
            raise NotImplementedError
        _logger.info('Written output to %s' % output_path, color='bold')

