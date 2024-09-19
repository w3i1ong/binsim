import os
import yaml
import argparse
from binsim.neural.lm.ins2vec import Ins2vec
from binsim.fs.corpus import CorpusDir
from binsim.disassembly.binaryninja.core import TokenCFGDataForm


def train_ins2vec(config):
    model_config, dataset_config = config['model'], config['dataset']
    model_kwargs = model_config['model_kwargs']
    model = Ins2vec(**model_kwargs)
    corpus_dir = CorpusDir(dataset_config['corpus'], dataset_config['name'], dataset_config['type'],
                           dataset_config['data-form'].value)
    save_dir = f'{corpus_dir.model_dir}/ins2vec'
    corpus_file_dir = corpus_dir.corpus_dir
    os.makedirs(save_dir, exist_ok=True)
    model.train(corpus_file_dir)
    model.save(os.path.join(save_dir, 'all-in-one.wv'))


def init_jtrans_parser(parser):
    pass


def init_palm_tree_parser(parser):
    pass


def init_bert_parser(parser):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='The configuration file for the model.')

    subparsers = parser.add_subparsers(dest='pretrained-model', required=True)
    ins2vec_parser = subparsers.add_parser('ins2vec')

    jtrans_parser = subparsers.add_parser('jtrans')

    palm_tree_parser = subparsers.add_parser('palm-tree')

    bert_parser = subparsers.add_parser('bert')

    return parser.parse_args()


def check_dataset_config(dataset_config):
    assert 'corpus' in dataset_config, 'The corpus of the dataset is not specified.'
    assert os.path.exists(
        dataset_config['corpus']), f"The corpus of the dataset does not exist: {dataset_config['corpus']}"
    assert 'type' in dataset_config, 'The type of the dataset is not specified.'
    dataset_type = dataset_config['type']
    assert dataset_type in ['InsCFG', 'TokenCFG'], f"Unknown dataset type: {dataset_type}, " \
                                                   f"supported types are {{InsCFG, TokenCFG}}."
    assert 'name' in dataset_config, 'The name of the dataset is not specified.'
    corpus_dir = os.path.join(dataset_config['corpus'], 'data', dataset_type, dataset_config['name'])
    assert os.path.exists(
        corpus_dir), f"There is no {dataset_type} corpus named {dataset_config['name']} in {dataset_config['corpus']}"

    data_form = TokenCFGDataForm(dataset_config.get('data-form', TokenCFGDataForm.InsStrSeq))
    dataset_config['data-form'] = data_form


def check_model_config(model_config):
    assert 'name' in model_config, 'The name of the model is not specified.'
    model_name = model_config['name']
    assert model_name in ['ins2vec', 'jtrans', 'palm-tree', 'bert'], f"Unknown model: {model_name}, " \
                                                                     f"supported models are {{ins2vec, jtrans, palm-tree, bert}}."
    model_config['train-per-arch'] = model_config.get('train-per-arch', False)
    model_config['model_kwargs'] = model_config.get('model_kwargs', {})


def load_and_check_config(config):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert 'dataset' in config, 'The dataset is not specified.'
    check_dataset_config(config['dataset'])

    assert 'model' in config, 'The model is not specified.'
    check_model_config(config['model'])
    return config


def main():
    args = parse_args()
    config = load_and_check_config(args.config)
    pretrained_model = config['model']['name']
    if pretrained_model == 'ins2vec':
        train_ins2vec(config)
    elif pretrained_model == 'jtrans':
        raise NotImplementedError()
    elif pretrained_model == 'palm-tree':
        raise NotImplementedError()
    elif pretrained_model == 'bert':
        raise NotImplementedError()
    else:
        raise RuntimeError(f"Unknown pretrained model: {pretrained_model}")


if __name__ == '__main__':
    main()
