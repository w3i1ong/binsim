import re
import os
import yaml
import pickle
import argparse
import logging
from typing import List, Union
from itertools import repeat
from multiprocessing.pool import Pool
from binsim.disassembly.binaryninja.core import BinaryNinja, PDGNormalizer, TokenCFGNormalizer, TokenCFGDataForm, \
    InsCFGNormalizer, InsCFG
from binsim.disassembly.binaryninja.core import TokenCFG, ProgramDependencyGraph
from tqdm import tqdm
from binsim.fs import CorpusDir, CorpusGraphType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s: %(message)s'))
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def list_corpus_binaries(corpus_dir):
    files = []
    for arch in os.listdir(corpus_dir):
        arch_dir = os.path.join(corpus_dir, arch)
        if os.path.isdir(arch_dir):
            for binary in os.listdir(arch_dir):
                binary_path = os.path.join(arch_dir, binary)
                if os.path.isfile(binary_path):
                    files.append(binary_path)
    return files


def generate_db_files(binaries, corpus_base, db_base):
    db_files = []
    for binary in binaries:
        db_file = os.path.join(db_base, os.path.relpath(binary, corpus_base) + '.bndb')
        db_files.append(db_file)
    return db_files


def generate_target_files(binaries, corpus_base, target_base):
    target_files = []
    for binary in binaries:
        target_files.append(os.path.join(target_base, os.path.relpath(binary, corpus_base)))
    return target_files


def check_disassembly_config(config):
    assert 'dataset' in config, 'The dataset field should be specified in the config file.'
    dataset_config = config['dataset']

    # check the correctness of the dataset config
    # the directory of binary files
    assert 'binary' in dataset_config, 'The binaries field should be specified in the config file.'
    assert os.path.exists(dataset_config['binary']), f'Cannot find binaries directory: {config["binary"]}'

    # the name of the corpus
    dataset_config['name'] = dataset_config.get('name', 'default')
    # the name must contain only letters, numbers, _ and -
    assert re.match(r'^[a-zA-Z0-9_-]+$', dataset_config['name']), \
        f'The name should contain only letters, numbers, _ and -, but got {dataset_config["name"]}.'

    # the directory to save our corpus
    assert 'corpus' in dataset_config, 'The corpus field should be specified in the config file.'
    if not os.path.exists(dataset_config['corpus']):
        os.makedirs(dataset_config['corpus'], exist_ok=True)
    assert os.path.isdir(dataset_config['corpus']), f'Cannot find corpus directory: {dataset_config["corpus"]}'

    # check the correctness of the disassembly config
    assert 'disassemble' in config, 'The dataset_config field should be specified in the config file.'
    disassemble_config = config['disassemble']
    # the type of graph to be generated
    assert 'type' in disassemble_config, 'The type field should be specified in the config file.'
    assert disassemble_config['type'] in [CorpusGraphType.PDG.value, CorpusGraphType.TokenCFG.value], \
        f'The type should be one of {{{CorpusGraphType.PDG.value}, {CorpusGraphType.TokenCFG.value}}}, ' \
        f'but got {disassemble_config["type"]}.'
    disassemble_config['type'] = CorpusGraphType(disassemble_config['type'])
    # How many processes to use
    disassemble_config['workers'] = disassemble_config.get('workers', 0)
    assert disassemble_config[
               'workers'] >= 0, f'The workers should be non-negative, but got {disassemble_config["workers"]}.'
    # the kwargs for the normalizer
    disassemble_config['normalizer_kwargs'] = disassemble_config.get('normalizer_kwargs', {})
    if disassemble_config['normalizer-kwargs'] is None:
        disassemble_config['normalizer-kwargs'] = {}
    else:
        if disassemble_config['type'] in [CorpusGraphType.TokenCFG]:
            normalizer_kwargs = disassemble_config['normalizer-kwargs']
            for key in list(normalizer_kwargs.keys()):
                if '-' in key:
                    new_key = key.replace('-', '_')
                    normalizer_kwargs[new_key] = normalizer_kwargs.pop(key)
        else:
            logger.warning("The normalizer-kwargs is not considered for PDG.")
    disassemble_config['verbose'] = disassemble_config.get('verbose', False)
    disassemble_config['keep-thunks'] = disassemble_config.get('keep-thunks', True)
    disassemble_config['keep-large'] = disassemble_config.get('keep-large', True)
    disassemble_config['large-ins-threshold'] = disassemble_config.get('large-ins-threshold', 1e10)
    disassemble_config['large-graph-threshold'] = disassemble_config.get('large-graph-threshold', 1e10)
    disassemble_config['keep-small'] = disassemble_config.get('keep-small', True)
    disassemble_config['small-ins-threshold'] = disassemble_config.get('small-ins-threshold', 0)
    disassemble_config['small-graph-threshold'] = disassemble_config.get('small-graph-threshold', 0)
    disassemble_config['regenerate'] = disassemble_config.get('regenerate', False)
    disassemble_config['reanalysis'] = disassemble_config.get('reanalysis', False)


def disassemble_corpus(config):
    # check the correctness of the config file
    check_disassembly_config(config)

    dataset_config = config['dataset']
    disassemble_config = config['disassemble']

    binary_dir = os.path.join(dataset_config['binary'])
    binaries = list_corpus_binaries(binary_dir)
    graph_type = CorpusGraphType(disassemble_config['type'])

    corpus_dir = CorpusDir(dataset_config['corpus'],
                           dataset_config['name'],
                           CorpusGraphType(disassemble_config['type']),
                           None)

    if graph_type == CorpusGraphType.PDG:
        normalizer_type = PDGNormalizer
    elif graph_type == CorpusGraphType.TokenCFG:
        normalizer_type = TokenCFGNormalizer
    else:
        raise ValueError(f'Unknown type: {graph_type}')

    db_files = generate_db_files(binaries, binary_dir, corpus_dir.database_dir)
    target_files = generate_target_files(binaries, binary_dir, corpus_dir.graph_dir)
    disassembler = BinaryNinja(normalizer_type, disassemble_config['normalizer-kwargs'])
    disassembler.disassemble_files(src_files=binaries,
                                   db_files=db_files,
                                   out_files=target_files,
                                   workers=disassemble_config['workers'],
                                   verbose=disassemble_config['verbose'],
                                   keep_thunks=disassemble_config['keep-thunks'],
                                   keep_unnamed=True,
                                   keep_large=disassemble_config['keep-large'],
                                   large_ins_threshold=disassemble_config['large-ins-threshold'],
                                   large_graph_threshold=disassemble_config['large-graph-threshold'],
                                   keep_small=disassemble_config['keep-small'],
                                   small_ins_threshold=disassemble_config['small-ins-threshold'],
                                   small_graph_threshold=disassemble_config['small-graph-threshold'],
                                   reanalysis=disassemble_config['reanalysis'],
                                   regenerate=disassemble_config['regenerate'])


def check_extract_dataset_subconfig(config):
    assert 'corpus' in config, 'The corpus field should be specified in the config file.'
    assert os.path.exists(config['corpus']) and os.path.isdir(config['corpus']), \
        f'Cannot find corpus directory: {config["corpus"]}'
    config['name'] = config.get('name', 'default')


def check_extract_subconfig(config):
    config['verbose'] = config.get('verbose', False)
    config['remove-duplicate'] = config.get('remove-duplicate', False)

    assert 'graph-type' in config, 'The Type field should be specified in the config file.'
    assert config['graph-type'] in ['TokenCFG', 'PDG'], f'Unknown graph-type: {config["corpus_type"]}'

    if config['graph-type'] == 'TokenCFG':
        config['workers'] = config.get('workers', 0)
        assert config['workers'] >= 0, f'The workers should be non-negative, but got {config["workers"]}.'

        config['type'] = CorpusGraphType(config.get('type', CorpusGraphType.TokenCFG.value))

        config['random-walk'] = config.get('random-walk', False)
        config['max-length'] = config.get('max-length', 100)
        config['walk-times'] = config.get('walk-times', 10)

        config['data-form'] = config.get('data-form', 'TokenSeq')
        config['data-form'] = TokenCFGDataForm(config['data-form'])

    elif config['graph-type'] == 'pdg':
        pass
    else:
        raise ValueError(f'Unknown type: {config["Type"]}')


def check_extract_config(config):
    assert 'dataset' in config, 'The dataset field should be specified in the config file.'
    dataset_config = config['dataset']
    check_extract_dataset_subconfig(dataset_config)

    assert 'extract' in config, 'The extract field should be specified in the config file.'
    extract_config = config['extract']
    check_extract_subconfig(extract_config)


def extract_sequence_from_token_cfg_wrapper(args):
    return extract_sequence_from_token_cfg(*args)


def extract_sequence_from_token_cfg(src_file,
                                    target_file,
                                    max_length=20,
                                    walk_times=10,
                                    data_form=TokenCFGDataForm.InsStrSeq):
    with open(src_file, 'rb') as f:
        token_cfg: List[Union[TokenCFG, InsCFG]] = pickle.load(f)

    target_dir = os.path.dirname(os.path.realpath(target_file))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    sequence = []
    for cfg in token_cfg:
        cfg.data_form = data_form
        random_walk = False
        if data_form in (TokenCFGDataForm.InsStrGraph, TokenCFGDataForm.TokenGraph):
            func_seq = cfg.random_walk(walk_num=walk_times, max_walk_length=max_length, random_walk=random_walk)
        else:
            func_seq = [cfg.as_token_list()]
        sequence.extend([(cfg.hash, seq) for seq in func_seq])
    with open(target_file, 'wb') as f:
        pickle.dump(sequence, f)


def get_cfg_files(token_cfg_dir: str) -> List[str]:
    token_cfg_files = []
    for arch in os.listdir(token_cfg_dir):
        arch_dir = os.path.join(token_cfg_dir, arch)
        for file in os.listdir(arch_dir):
            token_cfg_files.append(os.path.join(arch_dir, file))
    return token_cfg_files


def extract_sequence_for_corpus(corpus_dir,
                                workers=10,
                                token_type=CorpusGraphType.TokenCFG,
                                max_length=20,
                                walk_times=10,
                                data_form=TokenCFGDataForm.InsStrSeq,
                                remove_duplicate=True,
                                corpus_name='default', ):
    corpus_dir = CorpusDir(corpus_dir, corpus_name, token_type, data_form.value)
    cfg_files = get_cfg_files(corpus_dir.graph_dir)
    middle_files = [os.path.join(corpus_dir.cache_dir, os.path.relpath(file, corpus_dir.graph_dir)) for file in
                    cfg_files]
    args = zip(cfg_files, middle_files, repeat(max_length), repeat(walk_times), repeat(data_form))
    logger.info(f"Trying to extract sequences from {len(cfg_files)} token cfgs.")
    if workers == 0:
        _ = list(map(extract_sequence_from_token_cfg_wrapper, args))
    else:
        with Pool(workers) as pool:
            for _ in tqdm(pool.imap_unordered(extract_sequence_from_token_cfg_wrapper, args), total=len(cfg_files)):
                pass
    logger.info(f"Start to merge corpus.")
    cache_dir = corpus_dir.cache_dir
    for arch in os.listdir(cache_dir):
        logger.info(f"Processing corpus for {arch}.")
        outfile = corpus_dir.get_corpus_file(arch)
        middle_files = [os.path.join(cache_dir, arch, file) for file in os.listdir(os.path.join(cache_dir, arch))]
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, arch), exist_ok=True)
        middle_files = [file for file in middle_files if os.path.isfile(file)]
        with open(outfile, 'w') as out:
            if remove_duplicate:
                visited_func_hash = set()
                for file in tqdm(middle_files):
                    with open(file, 'rb') as f:
                        seq = pickle.load(f)
                    for func_hash, seq in seq:
                        if func_hash in visited_func_hash:
                            continue
                        visited_func_hash.add(func_hash)
                        out.write(' '.join(seq))
                        out.write('\n')
            else:
                for file in middle_files:
                    with open(file, 'rb') as f:
                        seq = pickle.load(f)
                    for _, seq in seq:
                        out.write(' '.join(seq))
                        out.write('\n')


def merge_pdg_for_corpus(corpus_dir,
                         corpus_name='default',
                         label='default',
                         remove_duplicate=True):
    corpus_dir = CorpusDir(corpus_dir, corpus_name, CorpusGraphType.PDG, label)
    cache_dir = corpus_dir.cache_dir
    for arch in os.listdir(cache_dir):
        outfile = corpus_dir.get_corpus_file(arch)
        pdg_files = [os.path.join(corpus_dir.cache_dir, arch, file) for file in
                     os.listdir(os.path.join(cache_dir, arch))]
        pdg_files = [file for file in pdg_files if os.path.isfile(file)]
        functions = []
        with open(outfile, 'wb') as out:
            if remove_duplicate:
                visited_func_hash = set()
                for file in pdg_files:
                    with open(file, 'rb') as f:
                        seq = pickle.load(f)
                    for function in seq:
                        if function.hash not in visited_func_hash:
                            visited_func_hash.add(function.hash)
                            functions.append(function)
            else:
                for file in pdg_files:
                    with open(file, 'rb') as f:
                        seq = pickle.load(f)
                    functions.extend(seq)
            pickle.dump(functions, out)


def extract_corpus(config):
    check_extract_config(config)
    dataset_config = config['dataset']
    extract_config = config['extract']
    if extract_config['graph-type'] == 'TokenCFG':
        graph_type = CorpusGraphType(extract_config['type'])
        extract_sequence_for_corpus(corpus_dir=dataset_config['corpus'],
                                    token_type=graph_type,
                                    workers=extract_config['workers'],
                                    max_length=extract_config['max-length'],
                                    walk_times=extract_config['walk-times'],
                                    remove_duplicate=extract_config['remove-duplicate'],
                                    corpus_name=dataset_config['name'],
                                    data_form=extract_config['data-form'])
    elif extract_config['corpus-type'] == 'pdg':
        merge_pdg_for_corpus(dataset_config['corpus'],
                             corpus_name=dataset_config['name'],
                             label=extract_config['name'],
                             remove_duplicate=extract_config['remove-duplicate'])
    else:
        raise ValueError(f'Unknown type: {config["Type"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='The configuration file.')
    subparsers = parser.add_subparsers(dest='action', required=True)
    disassemble_parser = subparsers.add_parser('disassemble')
    extract_parser = subparsers.add_parser('extract')
    args = parser.parse_args()

    config_file = args.config
    assert os.path.exists(config_file), f'Cannot find config file: {config_file}'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.action == 'disassemble':
        disassemble_corpus(config)
    elif args.action == 'extract':
        extract_corpus(config)


if __name__ == '__main__':
    main()
