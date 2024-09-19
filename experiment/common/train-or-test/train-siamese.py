import os
import random
import time
import yaml
import torch
import logging
import argparse
import datetime
from torch import nn
import _pickle as pickle
from typing import List, Tuple
from binsim.fs import DatasetDir
from torch.utils.data import DataLoader
from binsim.neural.nn.globals.siamese import *
from binsim.neural.nn.model import Gemini, I2vAtt, I2vRNN, RCFG2Vec, SAFE, AlphaDiff, GraphMatchingNet, Asteria, JTrans, ASTSAFE
from binsim.neural.nn.siamese import Siamese, SiameseMetric
from binsim.utils import get_optimizer_by_name
from binsim.neural.utils.data import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.utils import check_and_set_default
from tqdm import tqdm
from binsim.utils import load_pretrained_model

torch.multiprocessing.set_sharing_strategy('file_system')
logger = logging.getLogger("train-model")

class ModelSource(Enum):
    Recorded = "recorded"
    Pretrained = "pretrained"


def dict2str(d: dict) -> str:
    return ', '.join([f'{key}={value}' for key, value in d.items()])


def convert_kwargs_keys(kwargs):
    for key in list(kwargs):
        if '-' in key:
            kwargs[key.replace('-', '_')] = kwargs.pop(key)


def load_model(model_config, dataset_config, device: torch.device = 'cpu', sample_format=SiameseSampleFormat.Pair) \
        -> nn.Module:
    name = model_config['type']
    factory_kwargs = {'device': device, 'sample_format': sample_format}
    if name == 'Gemini':
        logger.info(f"The model to train is Gemini({dict2str(model_config['kwargs'])}).")
        return Gemini(**model_config['kwargs'], **factory_kwargs)
    elif name == 'i2v_rnn':
        logger.info(f"The model to train is i2v_rnn({dict2str(model_config['kwargs'])}).")
        return I2vRNN(**model_config['kwargs'], **factory_kwargs)
    elif name == 'i2v_att':
        logger.info(f"The model to train is i2v_att({dict2str(model_config['kwargs'])}).")
        return I2vAtt(**model_config['kwargs'], **factory_kwargs)
    elif name == 'SAFE':
        logger.info(f"The model to train is safe({dict2str(model_config['kwargs'])}).")
        return SAFE(**model_config['kwargs'], **factory_kwargs)
    elif name == 'RCFG2Vec':
        dataset_dir = DatasetDir(dataset_config['path'], graph_type='InsCFG', dataset_name=dataset_config['name'],
                                 merge_name=dataset_config['merge'])
        token2idx = dataset_dir.rel_to_data_dir('token2idx.pkl')
        with open(token2idx, 'rb') as f:
            token2idx = pickle.load(f)
        model_config['kwargs']['vocab_size'] = len(token2idx)
        logger.info(f"The model to train is RCFG2Vec({dict2str(model_config['kwargs'])}).")
        return RCFG2Vec(**model_config['kwargs'], **factory_kwargs)
    elif name == 'ASTSAFE':
        dataset_dir = DatasetDir(dataset_config['path'], graph_type='InsCFG', dataset_name=dataset_config['name'],
                                 merge_name=dataset_config['merge'])
        token2idx = dataset_dir.rel_to_data_dir('token2idx.pkl')
        with open(token2idx, 'rb') as f:
            token2idx = pickle.load(f)
        model_config['kwargs']['vocab_size'] = len(token2idx)
        logger.info(f"The model to train is ASTSAFE({dict2str(model_config['kwargs'])}).")
        return ASTSAFE(**model_config['kwargs'], **factory_kwargs)
    elif name == 'AlphaDiff':
        logger.info(f"The model to train is AlphaDiff({dict2str(model_config['kwargs'])}).")
        return AlphaDiff(**model_config['kwargs'], **factory_kwargs)
    elif name == 'GMN':
        logger.info(f"The model to train is GMN({dict2str(model_config['kwargs'])}).")
        return GraphMatchingNet(**model_config['kwargs'], **factory_kwargs)
    elif name == 'Asteria':
        logger.info(f"The model to train is Asteria({dict2str(model_config['kwargs'])}).")
        return Asteria(**model_config['kwargs'], **factory_kwargs)
    elif name == 'JTrans':
        logger.info(f"The model to train is JTrans.")
        return JTrans(pretrained_weights=model_config['kwargs']['pretrained_weights'])
    raise NotImplementedError(f"Unsupported model:{name}.")


def load_siamese(model, optimizer='Adam', device: torch.device = 'cpu', mixed_precision=False,
                 sample_format=SiameseSampleFormat.Pair) -> Siamese:
    logger.info(f"Initializing Siamese({model})...")
    optimizer = get_optimizer_by_name(optimizer)
    return Siamese(model, device=device, optimizer=optimizer,
                   mixed_precision=mixed_precision, sample_format=sample_format)


def parse_args():
    parser = argparse.ArgumentParser()
    # a config file which provide model-specific options
    parser.add_argument('--config',
                        type=str,
                        required=False,
                        help='The configuration file of the model.')
    sub_parsers = parser.add_subparsers(dest='action')
    sub_parsers.required = True
    train_parser = sub_parsers.add_parser('train')
    test_parser = sub_parsers.add_parser('test')
    return parser.parse_args()


def set_gpus(gpu_id):
    if not torch.cuda.is_available() or gpu_id is None:
        logger.info("No GPU is available, use CPU.")
        return torch.device('cpu')
    return torch.device(f'cuda:{gpu_id}')


def get_sample_dataset(graph_type: str, extra_options):
    from copy import deepcopy
    if extra_options is None:
        extra_options = {}
    for key in list(extra_options):
        if '-' in key:
            extra_options[key.replace('-', '_')] = extra_options.pop(key)

    sample_extra_kwargs, pair_extra_kwargs = deepcopy(extra_options), deepcopy(extra_options)
    if graph_type == 'ByteCode':  # Alpha-diff
        from binsim.neural.utils.data import ByteCodeSampleDataset, ByteCodeSamplePairDataset
        return ByteCodeSamplePairDataset, ByteCodeSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'ACFG':  # Gemini
        from binsim.neural.utils.data import ACFGSamplePairDataset, ACFGSampleDataset
        return ACFGSamplePairDataset, ACFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsStrCFG':  # i2v_att, i2v_rnn
        from binsim.neural.utils.data import InsStrCFGSamplePairDataset, InsStrCFGSampleDataset
        return InsStrCFGSamplePairDataset, InsStrCFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsStrSeq':  # SAFE
        from binsim.neural.utils.data import InsStrSeqSampleDataset, InsStrSeqSamplePairDataset
        return InsStrSeqSamplePairDataset, InsStrSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'CodeAST':  # Asteria
        from binsim.neural.utils.data import CodeASTSamplePairDataset, CodeASTSampleDataset
        return CodeASTSamplePairDataset, CodeASTSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'JTransSeq':  # JTrans
        from binsim.neural.utils.data import JTransSeqSampleDataset, JTransSeqSamplePairDataset
        return JTransSeqSamplePairDataset, JTransSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsDAG':  # RCFG2Vec
        from binsim.neural.utils.data import InsCFGSampleDataset, InsCFGSamplePairDataset
        return InsCFGSamplePairDataset, InsCFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'TokenDAG':  # RCFG2Vec(Token-Level)
        from binsim.neural.utils.data import TokenDAGSampleDataset, TokenDAGSamplePairDataset
        return TokenDAGSamplePairDataset, TokenDAGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsSeq':  # RCFG2Vec(Sequence model)
        from binsim.neural.utils.data import InsSeqSampleDataset, InsSeqSamplePairDataset
        return InsSeqSamplePairDataset, InsSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'MnemonicCFG':
        from binsim.neural.utils.data import InsStrCFGSamplePairDataset, InsStrCFGSampleDataset
        return InsStrCFGSamplePairDataset, InsStrCFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    else:
        raise NotImplementedError(f"Unsupported dataset type:{graph_type}.")


def load_train_data(dataset_config: dict) -> RandomSamplePairDatasetBase:
    if dataset_config['type'] in ['TokenStrCFG', 'TokenStrSeq', 'InsStrSeq', 'InsStrCFG']:
        data_type = 'TokenCFG'
    elif dataset_config['type'] in ['InsSeq', 'InsDAG', 'InsCFG']:
        data_type = 'InsCFG'
    else:
        data_type = dataset_config['type']
    graph_type = dataset_config['type']
    dataset_dir = DatasetDir(dataset_config['path'], dataset_name=dataset_config['name'], graph_type=data_type,
                             merge_name=dataset_config['merge'])

    pair_model, sample_model, pair_extra_kwargs, sample_extra_kwargs = \
        get_sample_dataset(graph_type, dataset_config['kwargs'])
    with open(dataset_dir.get_data_file(subset_name='train'),
              'rb') as f:
        print("Loading graphs...")
        train_data, data = pickle.load(f), {}
        function_names = set()
        for key in train_data:
            if len(train_data[key]) > 1:
                function_name = key.split(':')[-1]
                if function_name in function_names:
                    continue
                function_names.add(function_name)
                data[key] = list(train_data[key].values())
        pair_dataset = pair_model(data, **pair_extra_kwargs)
    return pair_dataset


def load_validation_data(dataset_config: dict,
                         search=False,
                         classification=False) -> Tuple[RandomSamplePairDatasetBase, List[SampleDatasetBase]]:
    if dataset_config['type'] in ['TokenSeq', 'TokenCFG', 'InsStrCFG', 'InsStrSeq', 'TokenDAG']:
        data_type = 'TokenCFG'
    elif dataset_config['type'] in ['InsSeq', 'InsDAG', 'InsCFG']:
        data_type = 'InsCFG'
    else:
        data_type = dataset_config['type']
    dataset_dir = DatasetDir(dataset_config['path'], graph_type=data_type, dataset_name=dataset_config['name'],
                             merge_name=dataset_config['merge'])
    graph_type = dataset_config['type']
    pair_model, sample_model, pair_extra_kwargs, sample_extra_kwargs = \
        get_sample_dataset(graph_type, dataset_config['kwargs'])

    with open(dataset_dir.get_data_file(subset_name='validation'), 'rb') as f:
        print("Loading validation graphs...")
        validation_data = pickle.load(f)
        search_data = classify_data = None

        if classification:
            data = {}
            for key in validation_data:
                data[key] = list(validation_data[key].values())
            classify_data = pair_model(data, **pair_extra_kwargs)

        if search:
            samples, names, tags = extract_samples_from_data(validation_data)
            name2id = {name: idx for idx, name in enumerate(set(names))}
            sample_id = torch.tensor([name2id[name] for name in names], dtype=torch.long)
            assert len(names) == len(samples)
            dataset = sample_model(samples, sample_id, tags=tags,
                                   with_name=True, **sample_extra_kwargs)
            search_data = [dataset, dataset]

    return classify_data, search_data

def get_ext_type(dataset_config: dict, dataset):
    dataset_dir = dataset_config['path']
    if dataset_config['type'] in ['TokenStrCFG', 'TokenStrSeq', 'InsStrSeq', 'InsStrCFG']:
        data_type = 'TokenCFG'
    elif dataset_config['type'] in ['InsSeq', 'InsDAG', 'InsCFG']:
        data_type = 'InsCFG'
    else:
        data_type = dataset_config['type']
    graph_type, dataset_name, merge_name = dataset_config['type'], dataset_config['name'], dataset_config['merge']
    dataset_dir = DatasetDir(dataset_dir, data_type, dataset_name=dataset_name, merge_name=merge_name)
    return sorted(os.listdir(dataset_dir.rel_to_data_dir(f'test/{dataset}')))

def get_datasets(dataset_config: dict):
    dataset_dir = dataset_config['path']
    if dataset_config['type'] in ['TokenStrCFG', 'TokenStrSeq', 'InsStrSeq', 'InsStrCFG']:
        data_type = 'TokenCFG'
    elif dataset_config['type'] in ['InsSeq', 'InsDAG', 'InsCFG']:
        data_type = 'InsCFG'
    else:
        data_type = dataset_config['type']
    graph_type, dataset_name, merge_name = dataset_config['type'], dataset_config['name'], dataset_config['merge']
    dataset_dir = DatasetDir(dataset_dir, data_type, dataset_name=dataset_name, merge_name=merge_name)
    return sorted(os.listdir(dataset_dir.rel_to_data_dir('test')))

def load_test_data(dataset_config: dict, dataset, exp_type) -> List[Tuple[SampleDatasetBase, SampleDatasetBase]]:
    dataset_dir = dataset_config['path']
    if dataset_config['type'] in ['TokenStrCFG', 'TokenStrSeq', 'InsStrSeq', 'InsStrCFG']:
        data_type = 'TokenCFG'
    elif dataset_config['type'] in ['InsSeq', 'InsDAG', 'InsCFG']:
        data_type = 'InsCFG'
    else:
        data_type = dataset_config['type']
    graph_type, dataset_name, merge_name = dataset_config['type'], dataset_config['name'], dataset_config['merge']
    dataset_dir = DatasetDir(dataset_dir, data_type, dataset_name=dataset_name, merge_name=merge_name)

    pair_model, sample_model, pair_extra_kwargs, sample_extra_kwargs = \
        get_sample_dataset(graph_type, dataset_config['kwargs'])

    test_data_list = []
    for file in os.listdir(dataset_dir.rel_to_data_dir(f'test/{dataset}/{exp_type}')):
        with open(dataset_dir.rel_to_data_dir(f'test/{dataset}/{exp_type}/{file}'), 'rb') as f:
            test_data = pickle.load(f)
        samples1, samples2 = test_data
        samples1, samples1_name, tags1 = zip(*samples1)
        samples2, samples2_name, tags2 = zip(*samples2)
        assert samples1_name == samples2_name

        samples_id1 = list(range(len(samples1_name)))
        samples_id2 = list(range(len(samples1_name)))
            # build DataSet Class
        sample_data1 = sample_model(samples1, samples_id1, with_name=True, tags=tags1, **sample_extra_kwargs)
        sample_data2 = sample_model(samples2, samples_id2, with_name=True, tags=tags2, **sample_extra_kwargs)
        search_data = (sample_data1, sample_data2)
        test_data_list.append(search_data)

    return test_data_list


def extract_samples_from_data(data, filter_func=None):
    samples, names, tags = [], [], []
    for idx, name in enumerate(data):
        functions = data[name]
        if filter_func is not None:
            functions = filter_func(name, functions)
        for options, function in functions.items():
            samples.append(function)
            names.append(name)
            tags.append(options)
    return samples, names, tags


def check_graph_type(model_name, config):
    assert 'type' in config, f"The graph type is not specified."
    graph_type = config['type']
    if model_name == 'Gemini':
        assert graph_type in ['ACFG'], f"Unsupported graph type:{graph_type} for Gemini."
    elif model_name in ['i2v_att', 'i2v_rnn']:
        raise NotImplementedError(f"{model_name} doesn't support graph type:{graph_type}.")
    elif model_name == 'RCFG2Vec':
        raise NotImplementedError(f"RCFG2Vec doesn't support graph type:{graph_type}.")
    elif model_name == 'safe':
        raise NotImplementedError(f"safe doesn't support graph type:{graph_type}.")
    raise NotImplementedError(f"Unsupported model:{model_name}.")


def check_model_config(config):
    assert 'type' in config, f"The model type is not specified."
    model_type = config['type']
    assert model_type in ['Gemini', 'i2v_att', 'i2v_rnn', 'RCFG2Vec', 'SAFE', 'ASTSAFE',
                          'AlphaDiff', 'Asteria', 'JTrans'], f"Model {model_type} is not supported now!"
    if 'kwargs' not in config or config['kwargs'] is None:
        config['kwargs'] = {}
    else:
        convert_kwargs_keys(config['kwargs'])
    logger.debug(f"Model Config: {config}")


def check_common_config(config):
    default_value = {'log': {'level': 'INFO', 'file': False}}
    required_kwargs = {'record-dir'}
    check_and_set_default('common-config', config, default_value, required_kwargs=required_kwargs, logger=logger)
    check_log_config(config['log'])


def check_dataset_config(config):
    required_kwargs = {'type', 'path', 'name'}
    default_values = {'kwargs': {}}
    check_and_set_default('dataset-config', config, default_value=default_values, required_kwargs=required_kwargs,
                          logger=logger)
    # todo: check whether the type is valid
    if not os.path.exists(config['path']):
        os.makedirs(config['path'], exist_ok=True)


def check_general_config_training(config):
    default_value = {
        'epoch': 50,
        'backward-steps': 1,
        'gpu': None,
        'mixed-precision': False,
        'num-workers': 10,
        'batch-size': 128,
        'choice-metric': SiameseMetric.AUC
    }
    check_and_set_default('train-general-config', config, default_value, logger=logger)
    config['choice-metric'] = SiameseMetric(config['choice-metric'])


def check_sampler_config_for_training(config):
    default_value = {
        'type': SiameseSampleFormat.Pair,
        'kwargs': {}
    }
    check_and_set_default('train-sampler-config', config, default_value, logger=logger)
    config['type'] = SiameseSampleFormat(config['type'])


def check_optimizer_config_for_training(config):
    default_value = {
        'lr-update-epoch': SiameseSampleFormat.Pair,
        'lr-update-scale': 0.9,
        'kwargs': {}
    }
    required_kwargs = {'type'}
    check_and_set_default('train-optimizer-config', config, default_value, required_kwargs=required_kwargs,
                          logger=logger)


def check_evaluation_config_for_training(config):
    default_value = {
        'val-interval': 5,
        'metrics': ['auc']
    }
    required_kwargs = {'batch-size'}
    check_and_set_default('train-evaluation-config', config, default_value, required_kwargs, logger=logger)

    if isinstance(config['batch-size'], int):
        batch_size = config['batch-size']
        config['batch-size'] = {'classification': batch_size, 'search': batch_size}
    else:
        check_and_set_default('train-evaluation-batch-size', config['batch-size'],
                              {'classification': 128, 'search': 256}, logger=logger)

    config['metrics'] = {SiameseMetric(metric) for metric in config['metrics']}


def check_training_config(config):
    assert 'general' in config, f"The general configuration is not specified for training."
    general_config = config['general']
    check_general_config_training(general_config)

    assert 'sampler' in config, f"The sampler is not specified for training."
    sampler_config = config['sampler']
    check_sampler_config_for_training(sampler_config)

    assert 'optimizer' in config, f"The optimizer is not specified for training."
    optimizer_config = config['optimizer']
    check_optimizer_config_for_training(optimizer_config)

    assert 'evaluation' in config, f"The evaluation is not specified for training."
    evaluation_config = config['evaluation']
    check_evaluation_config_for_training(evaluation_config)


def check_model_config_for_testing(config):
    default_value = {'model-source': ModelSource.Recorded}
    check_and_set_default('model-config', config, default_value=default_value, logger=logger)
    config['model-source'] = ModelSource(config['model-source'])

    if config['model-source'] == ModelSource.Pretrained:
        assert 'pretrained-weights' in config, f"The pretrained weights are not specified for testing."
        assert os.path.exists(
            config['pretrained-weights']), f"The pretrained weights doesn't exist:{config['pretrained-weights']}."


def check_general_config_for_testing(config):
    default_value = {'gpu': None, 'mixed-precision': False, 'num-workers': 10, 'pool-size': 1000, 'test-times': 10, 'random-seed': 0}
    required_value = {'batch-size'}
    check_and_set_default('test-general-config', config, default_value, required_value, logger=logger)
    if isinstance(config['batch-size'], int):
        batch_size = config['batch-size']
        config['batch-size'] = {'classification': batch_size, 'search': batch_size}
    else:
        check_and_set_default('test-general-config-batch-size', config['batch-size'],
                              {'classification': 128, 'search': 256}, logger=logger)


def check_evaluation_config_for_testing(config):
    default_value = {'metrics': ['auc']}
    required_kwargs = None
    check_and_set_default('test-evaluation-config', config, default_value,
                          required_kwargs=required_kwargs, logger=logger)
    config['metrics'] = {SiameseMetric(metric) for metric in config['metrics']}


def check_testing_config(config):
    required_kwargs = {'model', 'general', 'evaluation'}
    check_and_set_default('test-config', config, required_kwargs=required_kwargs, logger=logger)
    check_model_config_for_testing(config['model'])
    check_general_config_for_testing(config['general'])
    check_evaluation_config_for_testing(config['evaluation'])


def check_log_config(config):
    check_and_set_default('log-config', config, {'level': 'info'}, {'file'}, logger=logger)
    assert config['level'].upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], \
        f"Unknown log level:{config['level']}, supported log levels:{['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']}."
    config['level'] = config['level'].upper()
    config['file'] = config.get('file', False)


def check_config(config, train: bool):
    # check model name
    assert 'name' in config and config['name'] is not None, \
        "The name of this train/test is not specified."

    # check common config
    assert 'common' in config and config['common'] is not None, \
        "The common configuration(record-dir, log config) of this train/test is not specified."
    check_common_config(config['common'])

    # check model config
    assert 'model' in config, f"The model is not specified."
    check_model_config(config['model'])

    # check dataset config
    assert 'dataset' in config, f"The dataset is not specified."
    check_dataset_config(config['dataset'])

    assert 'train' in config, f"The training is not specified."
    training_config = config['train']
    check_training_config(training_config)

    assert 'test' in config, f"The test is not specified."
    testing_config = config['test']
    check_testing_config(testing_config)


def init_logger(level, file=False, path=None):
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s]: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if file or path is not None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        log_file = os.path.join(path, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)


def init_record_dir(path):
    os.makedirs(path, exist_ok=True)
    for subdir in ['log', 'train', 'test']:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)


def train_model(config, train: bool):
    # load config file
    assert os.path.exists(config), f"The config file doesn't exist:{config}."
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # check config
    check_config(config, train=train)
    common_config, model_config, dataset_config, train_config, test_config = \
        config['common'], config['model'], config['dataset'], config['train'], config['test']

    record_dir = common_config['record-dir']
    log_config = common_config['log']

    init_record_dir(record_dir)
    init_logger(**log_config, path=os.path.join(record_dir, 'log'))

    gpu = train_config['general']['gpu'] if train else test_config['general']['gpu']
    device = set_gpus(gpu)

    if device.type != 'cuda':
        logger.warning("Mixed precision is disabled as no GPU is available.")
        train_config['general']['mixed-precision'] = test_config['general']['mixed-precision'] = False

    if train:
        train_general_config, train_sampler_config, train_optimizer_config, train_evaluation_config = \
            train_config['general'], train_config['sampler'], train_config['optimizer'], train_config['evaluation']
        model = load_model(model_config, dataset_config, device=device, sample_format=train_sampler_config['type'])
        siamese = load_siamese(model, train_optimizer_config['type'], device=device,
                               mixed_precision=train_general_config['mixed-precision'],
                               sample_format=train_sampler_config['type'])
        logger.info(f"Loading train data from {dataset_config['path']}...")
        start_time = time.time()
        train_data = load_train_data(dataset_config)
        logger.info(f"Loading train data finished, time cost:{time.time() - start_time:.2f}s.")
        start_time = time.time()
        logger.info(f"Loading validation data from {dataset_config['path']}...")
        val_data, search_data = load_validation_data(dataset_config,
                                                     search=True,
                                                     classification=True)
        logger.info(f"Loading validation data finished, time cost:{time.time() - start_time:.2f}s.")
        logger.info(f"Start training, train config:")
        for key in train_general_config:
            logger.info(f"\t{key}={train_general_config[key]}")
        siamese.train(train_data, val_data,
                      search_data=search_data,
                      record_dir=record_dir,
                      backward_steps=train_general_config['backward-steps'],
                      epoch=train_general_config['epoch'],
                      val_interval=train_evaluation_config['val-interval'],
                      choice_metric=train_general_config['choice-metric'],
                      metrics=train_evaluation_config['metrics'],
                      lr=train_optimizer_config['lr'],
                      optimizer_kwargs=train_optimizer_config['kwargs'],
                      ignore_first=True,
                      eval_search_batch_size=train_evaluation_config['batch-size']['search'],
                      eval_classify_batch_size=train_evaluation_config['batch-size']['classification'],
                      train_batch_size=train_general_config['batch-size'],
                      num_workers=train_general_config['num-workers'],
                      lr_update_epoch=train_optimizer_config['lr-update-epoch'],
                      lr_update_scale=train_optimizer_config['lr-update-scale'])
    else:
        test_general_config, test_model_config, test_evaluation_config = \
            test_config['general'], test_config['model'], test_config['evaluation']

        # testing
        match test_model_config['model-source']:
            case ModelSource.Recorded:
                model_file = os.path.join(record_dir, 'model.pkl')
            case ModelSource.Pretrained:
                model_file = test_model_config['pretrained-weights']
            case _:
                raise ValueError(f"Unknown model source: {test_model_config['model-source']}")
        siamese_model = load_pretrained_model(model_config['type'], model_file, device=device)
        siamese = load_siamese(siamese_model, device=device, mixed_precision=test_general_config['mixed-precision'])

        logger.info(f"The parameter statistics of current model: {siamese.model.parameter_statistics}")
        os.makedirs(os.path.join(record_dir, 'test'), exist_ok=True)
        with open(os.path.join(record_dir, 'test', 'statistics.pkl'), 'wb') as f:
            pickle.dump(siamese.model.parameter_statistics, f)
        logger.info(f"Loading test data from {dataset_config['path']}...")
        num_workers = test_general_config['num-workers']
        eval_search_batch_size = test_general_config['batch-size']['search']
        for dataset in get_datasets(dataset_config):
            for comb_name in get_ext_type(dataset_config, dataset):
                test_results = {}
                test_data = load_test_data(dataset_config, dataset, comb_name)
                query_data, target_data = test_data[0]
                logger.info(
                    f"There are {len(query_data)} samples in query data and {len(target_data)} samples in target data.")
                for search_data in tqdm(test_data):
                    (query_data, target_data) = search_data
                    query_dataloader = DataLoader(query_data, batch_size=eval_search_batch_size, shuffle=True,
                                                  collate_fn=query_data.collate_fn_with_name, num_workers=num_workers)
                    target_dataloader = DataLoader(target_data, batch_size=eval_search_batch_size, shuffle=True,
                                                   collate_fn=target_data.collate_fn_with_name, num_workers=num_workers)
                    test_result = siamese.test(classify_loader=None,
                                               search_loader=(query_dataloader, target_dataloader),
                                               metrics=test_evaluation_config['metrics'],
                                               verbose=False,
                                               ignore_first=False)
                    for key, value in test_result.items():
                        if key not in test_results:
                            test_results[key] = []
                        test_results[key].append(value)
                logger.info(f"The test result for {comb_name} are,")
                for metric, metric_value in test_results.items():
                    logger.info(f"\t{metric}={sum(metric_value)/len(metric_value):.4f}")
                with open(f"{record_dir}/test/{comb_name}.pkl", 'wb') as f:
                    pickle.dump(test_results, f)


def main():
    init_logger(level='INFO')
    args = parse_args()
    train_model(args.config, train=(args.action == 'train'))


if __name__ == '__main__':
    main()
