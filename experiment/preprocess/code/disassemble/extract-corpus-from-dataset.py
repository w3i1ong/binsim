import yaml
import pickle
import argparse
from collections import defaultdict
from binsim.fs import CorpusDir, CorpusGraphType, DatasetDir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help="The path of the merge config.")
    return parser.parse_args()


def load_train_functions(dataset_path, dataset_name, merge_name):
    dataset = DatasetDir(dataset_path, graph_type="TokenCFG", dataset_name=dataset_name, merge_name=merge_name)
    with open(dataset.get_data_file("train"), 'rb') as f:
        cfgs = pickle.load(f)
    return cfgs


def extract_ins_freq(cfgs):
    ins2freq = defaultdict(lambda: 0)
    for func_name, options in cfgs.items():
        for option, cfg in options.items():
            tokens = cfg.as_token_list()
            for token in tokens:
                ins2freq[token] += 1
    return ins2freq


def generate_corpus(cfgs, outfile):
    with open(outfile, 'w') as corpus_file:
        for func_name, options in cfgs.items():
            for option, cfg in options.items():
                tokens = map(str, cfg.as_token_list())
                corpus_file.write(' '.join(tokens))
                corpus_file.write('\n')


def main():
    args = parse_args()
    config_file = args.config
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert config['dataset']['type'] == 'TokenCFG'
    dataset_path = config['dataset']['dataset']
    name = config['dataset']['name']
    merge_name = config['dataset']['merge_name']
    cfgs = load_train_functions(dataset_path, dataset_name=name, merge_name=merge_name)
    corpus_dir = config['dataset']['corpus']
    corpus_dir = CorpusDir(corpus_dir, name, CorpusGraphType.TokenCFG, "InsStrSeq", init=True)
    generate_corpus(cfgs, corpus_dir.get_corpus_file())


if __name__ == '__main__':
    main()
