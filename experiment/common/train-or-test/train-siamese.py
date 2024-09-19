import os, time, yaml, torch, random, logging, argparse
import _pickle as pickle
from torch import nn
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from binsim.neural.nn.globals.siamese import *
from binsim.neural.nn.model import (Gemini, I2vAtt, I2vRNN, RCFG2Vec, SAFE, AlphaDiff, GraphMatchingNet, Asteria,
                                    JTrans, ASTSAFE, BinMamba, CFGFormer)
from binsim.neural.nn.siamese import Siamese, SiameseMetric
from binsim.neural.utils.data import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.utils import (get_optimizer_by_name, get_sampler_by_name, get_distance_by_name, check_and_set_default,
                          load_pretrained_model, dict2str, init_logger)

torch.multiprocessing.set_sharing_strategy('file_system')
logger = logging.getLogger("train-model")

class ModelSource(Enum):
    Recorded = "recorded"
    Pretrained = "pretrained"
    Final = "final"
    Random = "random"

def convert_kwargs_keys(kwargs):
    for key in list(kwargs):
        if '-' in key:
            kwargs[key.replace('-', '_')] = kwargs.pop(key)


def load_model(model_config, device: torch.device = 'cpu') \
        -> nn.Module:
    name = model_config['type']
    model_kwargs = model_config['kwargs']
    distance_func = get_distance_by_name(model_config['distance']['type'], model_config['distance']['kwargs'])
    factory_kwargs = {'device': device, "distance_func": distance_func, **model_kwargs}
    logger.info(f"The model to train is {name}({factory_kwargs}).")
    if name == 'Gemini':
        return Gemini(**factory_kwargs)
    elif name == 'i2v_rnn':
        return I2vRNN(**factory_kwargs)
    elif name == 'i2v_att':
        return I2vAtt(**factory_kwargs)
    elif name == 'SAFE':
        return SAFE(**factory_kwargs)
    elif name == 'RCFG2Vec':
        return RCFG2Vec(**factory_kwargs)
    elif name == 'ASTSAFE':
        return ASTSAFE(**factory_kwargs)
    elif name == 'AlphaDiff':
        return AlphaDiff(**factory_kwargs)
    elif name == 'GMN':
        return GraphMatchingNet(**model_config['kwargs'], **factory_kwargs)
    elif name == 'Asteria':
        return Asteria(**factory_kwargs)
    elif name == 'JTrans':
        return JTrans(pretrained_weights=model_config['kwargs']['pretrained_weights'])
    elif name == 'BinMamba':
        return BinMamba(**factory_kwargs)
    elif name == "CFGFormer":
        return CFGFormer(**factory_kwargs)
    raise NotImplementedError(f"Unsupported model:{name}.")


def load_siamese(model, optimizer='Adam', device: torch.device = 'cpu',
                 sampler_config=None, momentum_model=None) -> Siamese:
    logger.info(f"Initializing Siamese({model})...")
    optimizer = get_optimizer_by_name(optimizer)
    sampler, sample_format = None, None
    if sampler_config is not None:
        if sampler_config["type"] is not None:
            sampler = get_sampler_by_name(sampler_config["type"], sampler_config["kwargs"])
        if sampler_config["dataset-sample-format"] is not None:
            sample_format = SiameseSampleFormat(sampler_config["dataset-sample-format"])

    return Siamese(model, device=device, optimizer=optimizer,
                   sample_format=sample_format,
                   sampler=sampler, momentum_model=momentum_model)


def parse_args():
    parser = argparse.ArgumentParser()
    # a config file which provide model-specific options
    parser.add_argument('--basic-config',
                        type=str,
                        required=False,
                        help='The configuration file of the model.')
    sub_parsers = parser.add_subparsers(dest='action')
    sub_parsers.required = True
    train_parser = sub_parsers.add_parser('train')
    test_parser = sub_parsers.add_parser('test')
    test_parser.add_argument("--test-config", type=str, required=True,
                             help="The configuration file for testing.")
    args = parser.parse_args()
    with open(args.basic_config, 'r') as f:
        base_config = yaml.safe_load(f)
    if args.action == 'test':
        with open(args.test_config, 'r') as f:
            test_config = yaml.safe_load(f)
        base_config["test"] = test_config
    return base_config, args.action



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
    elif graph_type == 'TokenCFG':  # i2v_att, i2v_rnn
        from binsim.neural.utils.data import TokenCFGSamplePairDataset, TokenCFGSampleDataset
        return TokenCFGSamplePairDataset, TokenCFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'TokenSeq':  # SAFE
        from binsim.neural.utils.data import TokenSeqSampleDataset, TokenSeqSamplePairDataset
        return TokenSeqSamplePairDataset, TokenSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'CodeAST':  # Asteria
        from binsim.neural.utils.data import CodeASTSamplePairDataset, CodeASTSampleDataset
        return CodeASTSamplePairDataset, CodeASTSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'JTransSeq':  # JTrans
        from binsim.neural.utils.data import JTransSeqSampleDataset, JTransSeqSamplePairDataset
        return JTransSeqSamplePairDataset, JTransSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsCFG':  # RCFG2Vec
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

def remove_small_functions(graph_db, threshold):
    removed_graph, removed_function = 0, 0
    for graph_name in list(graph_db.keys()):
        is_small_graph = True
        for option, function in graph_db[graph_name].items():
            if function.node_num >= threshold:
                is_small_graph = False
                break
        if is_small_graph:
            removed_graph += len(graph_db[graph_name])
            removed_function += 1
            del graph_db[graph_name]
    return removed_function, removed_function


def load_train_data(dataset_config: dict) -> RandomSamplePairDatasetBase:
    graph_type = dataset_config['type']
    pair_model, sample_model, pair_extra_kwargs, sample_extra_kwargs = \
        get_sample_dataset(graph_type, dataset_config['kwargs'])
    small_graph_threshold = dataset_config["small-graph-threshold"]
    with open(os.path.join(dataset_config['path'], 'train', 'meta.pkl'), 'rb') as f:
        print("Loading graphs...")
        train_data, data = pickle.load(f), {}
        removed_func, removed_graph = remove_small_functions(train_data, small_graph_threshold)
        logger.info(f"{removed_func} functions({removed_graph} graphs) are removed due to the small graph size.")
        function_names = set()
        for key in train_data:
            if len(train_data[key]) > 1:
                function_name = key.split(':')[-1]
                if function_name in function_names:
                    continue
                function_names.add(function_name)
                data[key] = list(train_data[key].values())
        pair_dataset = pair_model(data, **pair_extra_kwargs, neural_input_cache_rocks_file=os.path.join(dataset_config['path'], 'train', 'dataset.db'))
    return pair_dataset


def load_validation_data(dataset_config: dict) -> List[SampleDatasetBase]:
    dataset_dir = dataset_config['path']
    graph_type = dataset_config['type']
    pair_model, sample_model, pair_extra_kwargs, sample_extra_kwargs = \
        get_sample_dataset(graph_type, dataset_config['kwargs'])
    small_graph_threshold = dataset_config["small-graph-threshold"]
    with open(os.path.join(dataset_dir, "validation", "meta.pkl"), 'rb') as f:
        print("Loading validation graphs...")
        validation_data = pickle.load(f)
        removed_func, removed_graph = remove_small_functions(validation_data, small_graph_threshold)
        logger.info(f"{removed_func} functions({removed_graph} graphs) are removed due to the small graph size.")
        samples, names, tags = extract_samples_from_data(validation_data)
        name2id = {name: idx for idx, name in enumerate(set(names))}
        sample_id = torch.tensor([name2id[name] for name in names], dtype=torch.long)
        assert len(names) == len(samples)
        dataset = sample_model(samples, sample_id, tags=tags,
                               with_name=True, **sample_extra_kwargs, neural_input_cache_rocks_file=os.path.join(dataset_dir, 'validation', 'dataset.db'))
        search_data = [dataset, dataset]

    return search_data

def get_datasets(dataset_config: dict):
    dataset_dir = dataset_config['path']
    return sorted(os.listdir(dataset_dir.rel_to_data_dir('test')))

def build_search_data(data, option1, option2, *, pool_size, test_time, random_seed,
                      graph_type, cache_file, sample_extra_kwargs):
    option1, option2 = tuple(option1.split(':')), tuple(option2.split(':'))
    functions = []
    _, sample_dataset, _, sample_extra_kwargs = get_sample_dataset(graph_type, sample_extra_kwargs)
    for function_name, graphs in data.items():
        if option1 in graphs and option2 in graphs:
            functions.append((graphs[option1], (function_name, option1),
                              graphs[option2], (function_name, option2)))
    assert pool_size=='all' or len(functions) >= pool_size, f"The number of functions({len(functions)}) is less than the pool size({pool_size})."
    random.seed(random_seed)
    for _ in range(test_time):
        random.shuffle(functions)
        if pool_size != 'all':
            samples1, tags1, samples2, tags2 = zip(*functions[:pool_size])
            samples_id1, samples_id2 = list(range(pool_size)), list(range(pool_size))
        else:
            samples1, tags1, samples2, tags2 = zip(*functions)
            samples_id1, samples_id2 = list(range(len(functions))), list(range(len(functions)))
            graph_size = [0 for _ in range(11)]
            for graph in samples1:
                graph_size[graph.node_num // 30] += 1
            # build DataSet Class
        sample_data1 = sample_dataset(samples1, samples_id1, with_name=True, tags=tags1, neural_input_cache_rocks_file=cache_file, **sample_extra_kwargs)
        sample_data2 = sample_dataset(samples2, samples_id2, with_name=True, tags=tags2, neural_input_cache_rocks_file=cache_file, **sample_extra_kwargs)
        yield sample_data1, sample_data2



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
                          'AlphaDiff', 'Asteria', 'JTrans', 'BinMamba', "CFGFormer" ], f"Model {model_type} is not supported now!"
    if 'kwargs' not in config or config['kwargs'] is None:
        config['kwargs'] = {}
    else:
        convert_kwargs_keys(config['kwargs'])
    assert "distance" in config, f"The distance function is not specified."
    assert 'type' in config['distance'], f"The distance function type is not specified."
    config['distance']['kwargs'] = config['distance'].get('kwargs', {})
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
        'num-workers': 10,
        'batch-size': 128,
        'choice-metric': SiameseMetric.nDCG(10),
        "use-momentum-model": False
    }
    check_and_set_default('train-general-config', config, default_value, logger=logger)
    config['choice-metric'] = SiameseMetric(config['choice-metric'])


def check_sampler_config_for_training(config):
    default_value = {
        'type': None,
        'kwargs': {}
    }
    check_and_set_default('train-sampler-config', config,
                          default_value, required_kwargs={"dataset-sample-format"},
                          logger=logger)
    config['dataset-sample-format'] = SiameseSampleFormat(config['dataset-sample-format'])


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

    assert isinstance(config["batch-size"], int) and config["batch-size"] > 0, \
        f"Invalid batch size:{config['batch-size']}."

    config['metrics'] = {SiameseMetric(metric) for metric in config['metrics']}

def check_training_config(config):
    assert 'general' in config, f"The general configuration is not specified for training."
    general_config = config['general']
    check_general_config_training(general_config)

    assert 'sampler' in config, f"The sampler is not specified for training."
    sampler_config = config['sampler']
    check_sampler_config_for_training(sampler_config)

    assert "loss" in config, f"Loss function should be specified for training."
    loss_config = config['loss']

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
    default_value = {'gpu': None, 'num-workers': 10, 'pool-size': 1000, 'test-times': 10, 'random-seed': 0}
    required_value = {'batch-size'}
    check_and_set_default('test-general-config', config, default_value, required_value, logger=logger)
    assert isinstance(config['batch-size'], int) and config['batch-size'] > 0, \
        "Batch size should be a positive integer, but got {config['batch-size']}."


def check_evaluation_config_for_testing(config):
    default_value = {'metrics': ['ndcg@10']}
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
    if train:
        assert 'train' in config, f"The training is not specified."
        training_config = config['train']
        check_training_config(training_config)
    else:
        assert 'test' in config, f"The test is not specified."
        testing_config = config['test']
        check_testing_config(testing_config)


def init_record_dir(path):
    os.makedirs(path, exist_ok=True)
    for subdir in ['log', 'train', 'test']:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)


def train_model(config, train: bool):
    # check config
    check_config(config, train=train)
    if train:
        common_config, model_config, dataset_config, train_config = \
            config['common'], config['model'], config['dataset'], config['train']

        record_dir = common_config['record-dir']
        log_config = common_config['log']

        init_record_dir(record_dir)
        log_file = os.path.join(record_dir, 'log', 'train-siamese') if log_config["file"] else None
        init_logger("train-siamese", level=log_config["level"], file=log_file)

        gpu = train_config['general']['gpu']
        device = set_gpus(gpu)

        train_general_config, train_sampler_config, train_optimizer_config, train_evaluation_config = \
            train_config['general'], train_config['sampler'], train_config['optimizer'], train_config['evaluation']
        train_loss_config = train_config['loss']
        model = load_model(model_config, device=device)

        momentum_model = None
        if train_general_config["use-momentum-model"]:
            momentum_model = load_model(model_config, device=device)

        siamese = load_siamese(model, optimizer=train_optimizer_config['type'], device=device,
                               sampler_config=train_sampler_config, momentum_model=momentum_model)

        logger.info(f"Loading train data from {dataset_config['path']}...")
        start_time = time.time()
        train_data = load_train_data(dataset_config)
        logger.info(f"Loading train data finished, time cost:{time.time() - start_time:.2f}s.")
        start_time = time.time()
        logger.info(f"Loading validation data from {dataset_config['path']}...")
        search_data = load_validation_data(dataset_config)
        logger.info(f"Loading validation data finished, time cost:{time.time() - start_time:.2f}s.")
        logger.info(f"Start training, train config:")
        for key in train_general_config:
            logger.info(f"\t{key}={train_general_config[key]}")
        siamese.train(train_data,
                      search_data=search_data,
                      record_dir=record_dir,
                      backward_steps=train_general_config["backward-steps"],
                      epoch=train_general_config["epoch"],
                      val_interval=train_evaluation_config["val-interval"],
                      choice_metric=train_general_config["choice-metric"],
                      metrics=train_evaluation_config["metrics"],
                      lr=train_optimizer_config["lr"],
                      loss_func_name=train_loss_config["type"],
                      loss_func_kwargs=train_loss_config["kwargs"],
                      optimizer_kwargs=train_optimizer_config["kwargs"],
                      ignore_first=True,
                      queue_max_size=train_general_config["queue-max-size"],
                      eval_search_batch_size=train_evaluation_config["batch-size"],
                      train_batch_size=train_general_config["batch-size"],
                      num_workers=train_general_config["num-workers"],
                      lr_update_epoch=train_optimizer_config["lr-update-epoch"],
                      lr_update_scale=train_optimizer_config["lr-update-scale"])
    else:
        common_config, model_config, dataset_config, test_config = \
            config['common'], config['model'], config['dataset'], config['test']
        test_general_config, test_model_config, test_evaluation_config = \
            test_config['general'], test_config['model'], test_config['evaluation']
        record_dir = common_config['record-dir']
        log_config = common_config['log']

        init_record_dir(record_dir)
        log_file = os.path.join(record_dir, 'log', 'train-siamese') if log_config["file"] else None
        init_logger("train-siamese", level=log_config["level"], file=log_file)

        gpu = test_config['general']['gpu']
        device = set_gpus(gpu)

        # testing
        match test_model_config['model-source']:
            case ModelSource.Recorded:
                model_file = os.path.join(record_dir, 'model.pkl')
            case ModelSource.Final:
                model_file = os.path.join(record_dir, 'latest.pkl')
            case ModelSource.Pretrained:
                model_file = test_model_config['pretrained-weights']
            case ModelSource.Random:
                model_file = None
            case _:
                raise ValueError(f"Unknown model source: {test_model_config['model-source']}")
        if model_file:
            siamese_model = load_pretrained_model(model_config['type'], model_file, device=device)
        else:
            siamese_model = load_model(model_config, device=device)
        siamese = load_siamese(siamese_model, device=device)

        logger.info(f"The parameter statistics of current model: {siamese.model.parameter_statistics}")
        os.makedirs(os.path.join(record_dir, 'test'), exist_ok=True)
        with open(os.path.join(record_dir, 'test', 'statistics.pkl'), 'wb') as f:
            pickle.dump(siamese.model.parameter_statistics, f)
        logger.info(f"Loading test data from {dataset_config['path']}...")
        num_workers = test_general_config['num-workers']
        eval_search_batch_size = test_general_config['batch-size']
        with open(os.path.join(dataset_config['path'], 'test', 'meta.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        options = test_evaluation_config["options"]
        pool_size = test_general_config['pool-size']
        random_seed = test_general_config['random-seed']
        test_time = test_general_config['test-times']
        cache_file = os.path.join(dataset_config['path'], 'test', 'dataset.db')
        perf_record = {}
        for option_name, (option1, option2) in options.items():
            test_results = {}
            for query, target in tqdm(build_search_data(test_data, option1, option2, pool_size=pool_size,
                                                        test_time=test_time, random_seed=random_seed,
                                                        cache_file=cache_file,
                                                        graph_type=dataset_config['type'],
                                                        sample_extra_kwargs=dataset_config['kwargs']), total=test_time):
                query_dataloader = DataLoader(query, batch_size=eval_search_batch_size, shuffle=True,
                                              collate_fn=query.collate_fn_with_name, num_workers=num_workers)
                target_dataloader = DataLoader(target, batch_size=eval_search_batch_size, shuffle=True,
                                               collate_fn=target.collate_fn_with_name, num_workers=num_workers)
                test_result = siamese.test(search_loader=(query_dataloader, target_dataloader),
                                           metrics=test_evaluation_config['metrics'],
                                           verbose=False,
                                           ignore_first=False)
                for key, value in test_result.items():
                    if key not in test_results:
                        test_results[key] = []
                    test_results[key].append(value)
            print(test_results)
            perf_record[option_name] = test_results
            with open(os.path.join(record_dir, 'test', 'perf_record.pkl'), 'wb') as f:
                pickle.dump(perf_record, f)

def main():
    init_logger(logger_name="train-model", level='INFO')
    config, action = parse_args()
    train_model(config, train=(action == 'train'))


if __name__ == '__main__':
    main()
