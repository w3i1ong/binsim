import os
import yaml
import argparse
from binsim.neural.lm.ins2vec import Ins2vec


def train_word2vec(config):
    model_config = config['model']
    model_kwargs = model_config['model_kwargs']
    corpus, model_file = model_config["corpus"], model_config["model-file"]
    save_dir = os.path.dirname(model_file)
    model = Ins2vec(**model_kwargs)
    os.makedirs(save_dir, exist_ok=True)
    model.train(corpus)
    model.save(model_file)


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


def check_model_config(model_config):
    assert 'name' in model_config, 'The name of the model is not specified.'
    model_name = model_config['name']
    assert model_name in ['ins2vec', 'jtrans', 'palm-tree', 'bert'], f"Unknown model: {model_name}, " \
                                                                     f"supported models are {{ins2vec, jtrans, palm-tree, bert}}."
    model_config['model_kwargs'] = model_config.get('model_kwargs', {})
    assert "corpus" in model_config, 'The corpus is not specified.'
    assert os.path.exists(model_config["corpus"]), f"The corpus file {model_config['corpus']} does not exist."
    assert "model-file" in model_config, 'The model file is not specified.'


def load_and_check_config(config):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert 'model' in config, 'The model is not specified.'
    check_model_config(config['model'])
    return config


def main():
    args = parse_args()
    config = load_and_check_config(args.config)
    pretrained_model = config['model']['name']
    if pretrained_model == 'ins2vec':
        train_word2vec(config)
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
