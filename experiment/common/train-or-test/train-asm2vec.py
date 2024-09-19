import os
import yaml
import torch
import logging
import argparse
import pickle
from binsim.neural.lm import Asm2Vec
from binsim.neural.lm.asm2vec import NamedTokenCFGs
from binsim.fs import DatasetDir, RecordDir
from binsim.neural.nn.metric import search
from binsim.neural.nn.distance import PairwiseCosineDistance
from binsim.neural.nn.siamese import Siamese, SiameseMetric
from binsim.utils import init_logger as set_logger
logger = set_logger("train-model", logging.DEBUG, console=True, console_level=logging.DEBUG)

def init_logger(config):
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()
    set_logger(logger.name, config['level'], console=config['console'], console_level=config['level'])


def parse_args():
    parser = argparse.ArgumentParser(description='Train asm2vec model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    subparser = parser.add_subparsers(dest='action', help='Mode of training.', required=True)
    subparser.add_parser('train', help='Train a new model.')
    subparser.add_parser('test', help='Test the performance of a trained model.')
    return parser.parse_args()


def check_dataset_config(config: dict):
    assert 'type' in config, 'Type is not provided in the dataset config.'
    assert config['type'] == 'TokenCFG', 'Only TokenCFG is supported for asm2vec!'
    assert 'path' in config, 'Path is not provided in the dataset config.'
    assert os.path.exists(config['path']), f'Dataset path {config["path"]} does not exist.'
    assert 'name' in config, 'Name for dataset is not specified in dataset config.'


def check_log_config(config: dict):
    assert 'level' in config, 'Level is not specified in the log config.'
    config['level'] = config['level'].upper()
    assert config['level'].upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], \
        f'Level {config["level"]} is not supported.'
    config['console'] = config.get('console', True)
    init_logger(config)


def check_common_config(config: dict):
    assert 'record-dir' in config, 'Record directory is not specified in the common config.'
    os.makedirs(config['record-dir'], exist_ok=True)
    config['num-workers'] = config.get('num-workers', 10)
    config['max-walk-path'] = config.get('max-walk-path', 10000)
    config['random-walk-num'] = config.get('random-walk-num', 10)


def check_config(config: dict, action: str):
    # check name
    assert 'name' in config, 'Name is not provided in the config file.'
    # check dataset
    assert 'dataset' in config, 'Dataset is not provided in the config file.'
    check_dataset_config(config['dataset'])

    # check log
    assert 'log' in config, 'Log is not provided in the config file.'
    check_log_config(config['log'])

    # check common
    assert 'common' in config, 'Common is not provided in the config file.'
    check_common_config(config['common'])

    # check train
    assert "train" in config, "Train Config is not provided."
    check_train_config(config['train'])

    # check test
    assert "test" in config, "Test Config is not provided."
    check_test_config(config['test'])


def check_train_config(config: dict):
    assert 'epoch' in config, 'Epoch is not provided in the train config.'
    assert 'embed-dim' in config, 'Embedding dimension is not provided in the train config.'
    config['min-count'] = config.get('min-count', 0)
    config['window-size'] = config.get('window-size', 1)
    assert 'lr' in config, 'Learning rate is not provided in the train config.'


def check_test_config(config: dict):
    assert 'epoch' in config, 'Epoch is not provided in the test config.'
    assert 'metrics' in config, 'Metrics is not provided in the test config.'
    assert 'combinations' in config, 'Combination is not provided in the test config.'


def train_asm2vec(config):
    dataset_config, train_config, common_config = config['dataset'], config['train'], config['common']
    name = config['name']
    dataset_dir = DatasetDir(dataset_config['path'])
    graph_type = dataset_config['type']
    dataset_name = dataset_config['name']
    train_data = dataset_dir.get_data_file(graph_type, 'train', dataset_name)

    record_dir = RecordDir(common_config['record-dir'], name)
    logger.info("Loading training data.")
    # load data
    with open(train_data, 'rb') as f:
        cfgs = pickle.load(f)

    graph_list, names = [], []
    for name in cfgs:
        graph_list.extend(cfgs[name].values())
        names.extend([name] * len(cfgs[name]))
    graph_list = graph_list
    logger.info(f"Training data loaded, there are {len(graph_list)} graphs in total.")
    # start training
    logger.info("Start training asm2vec model.")
    asm2vec = Asm2Vec(workers=common_config['num-workers'],
                      embed_dim=train_config['embed-dim'],
                      min_count=train_config['min-count'],
                      window_size=train_config['window-size'],
                      epoch=train_config['epoch'],
                      logger=logger)
    dataset = NamedTokenCFGs(graph_list, names=names)
    asm2vec.train(dataset)
    asm2vec.save(record_dir.model_file)


def test_asm2vec(config):
    dataset_config, train_config, common_config = config['dataset'], config['train'], config['common']
    name = config['name']
    dataset_dir = DatasetDir(dataset_config['path'])
    graph_type = dataset_config['type']
    dataset_name = dataset_config['name']
    test_data = dataset_dir.get_data_file(graph_type, 'test', dataset_name)
    record_dir = RecordDir(common_config['record-dir'], name)
    logger.info("Loading test data.")
    # load data
    with open(test_data, 'rb') as f:
        cfgs = pickle.load(f)

    combinations = config['test']['combinations']
    metrics = set([SiameseMetric(metric) for metric in config['test']['metrics']])
    model = Asm2Vec.load(record_dir.model_file)

    for comb_name, combination in combinations.items():
        # generate search data
        combination1, combination2 = combination[0], combination[1]
        combination1, combination2 = tuple(combination1.split(':')), tuple(combination2.split(':'))
        graphs1, graphs2, names = [], [], []
        for name in cfgs:
            comb2graph = cfgs[name]
            if combination1 in comb2graph and combination2 in comb2graph:
                graphs1.append(comb2graph[combination1])
                graphs2.append(comb2graph[combination2])
                names.append(name)

        logger.info(f"Test data loaded, there are {len(graphs1)} graphs in total.")

        # start training
        logger.info("Start testing asm2vec model.")
        name2idx1, embeddings1 = model.generate_embedding(NamedTokenCFGs(graphs1, names), epochs=1)
        name2idx2, embeddings2 = model.generate_embedding(NamedTokenCFGs(graphs2, names), epochs=1)
        ids1 = torch.tensor([name2idx1[name] for name in names])
        ids2 = torch.tensor([name2idx2[name] for name in names])
        pairwise_cosine_distance = PairwiseCosineDistance()
        similarity_func = lambda x, y: 1 - pairwise_cosine_distance(x, y)
        embeddings1, embeddings2 = torch.from_numpy(embeddings1), torch.from_numpy(embeddings2)
        search_result, answer = search(embeddings1, ids1, embeddings2, ids2, top_k=100, pair_sim_func=similarity_func,
                                       device='cpu')

        search_result_correct = (ids1.reshape([len(ids1), 1]) == ids2[search_result]).float()
        search_results = Siamese.calculate_search_metrics(search_result_correct, answer, metrics, ignore_first=False)

        logger.info(f"The test result for {combination1} vs {combination2} are,")
        for metric, metric_value in search_results.items():
            logger.info(f"\t{metric}={metric_value}")
        with open(f"{record_dir.test_record}/{comb_name}.pkl", 'wb') as f:
            pickle.dump(search_results, f)


def main():
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    check_config(config, args.action)
    if args.action == 'train':
        train_asm2vec(config)
    else:
        test_asm2vec(config)


if __name__ == '__main__':
    main()
