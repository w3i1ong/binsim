import os
import random
import tqdm
import yaml
import pickle
import logging
import argparse
from typing import List, Union
from binsim.disassembly.utils.globals import GraphType
from binsim.utils import dict2str, init_logger, check_and_set_default
from collections import Counter, defaultdict
from binsim.fs import DatasetDir

logger = init_logger(os.path.basename(__file__), logging.INFO,
                     console=True, console_level=logging.INFO)


def list_dataset_binaries(dataset: str, arch: List[str], compiler: List[str]) -> List[str]:
    binaries = []
    for software in os.listdir(dataset):
        software_dir = os.path.join(dataset, software)
        for version in os.listdir(software_dir):
            version_dir = os.path.join(software_dir, version)
            for _arch in os.listdir(version_dir):
                if _arch not in arch:
                    continue
                arch_dir = f"{version_dir}/{_arch}"
                for _os in os.listdir(arch_dir):
                    os_dir = f"{arch_dir}/{_os}"
                    for _compiler in os.listdir(os_dir):
                        if compiler is not None and _compiler not in compiler:
                            continue
                        compiler_dir = f"{os_dir}/{_compiler}"
                        for root, _, files in os.walk(compiler_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                # skip symbolic link
                                if os.path.islink(file_path):
                                    continue
                                binaries.append(file_path)
    return binaries


def check_dataset_config(config: dict):
    check_and_set_default('dataset-config', config,
                          default_value={'kwargs': {}},
                          required_kwargs={'name', 'binary', 'dataset', 'type'}, logger=logger)
    assert os.path.exists(config['binary']), f'The directory of original binaries({config["binary"]}) does not exist.'
    if config['kwargs'] is None:
        config['kwargs'] = {}
    config['type'] = GraphType(config['type'])


def check_disassemble_config(config: dict):
    check_and_set_default("disassemble-config",
                          kwargs=config,
                          default_value={
                              'workers': 10, 'verbose': False, 'keep-thunks': False, 'keep-large': False,
                              'regenerate': False, 'reanalysis': False, 'large-ins-threshold': 1000000,
                              'large-graph-threshold': 700, 'keep-small': False, 'small-ins-threshold': 10,
                              'small-graph-threshold': 5, 'kwargs': {},
                              'compiler': None
                          },
                          required_kwargs={'arch'},
                          logger=logger)
    assert isinstance(config['workers'], int), \
        f'The number of workers should be an integer, but got {type(config["workers"])}.'
    assert config['workers'] >= 0, 'The number of workers should not be greater than 0, but got {config["workers"]}.'
    check_support_arch(config['arch'])
    config['arch'] = set(config['arch'])
    config['compiler'] = set(config['compiler']) if config['compiler'] is not None else None


def check_support_arch(arch_set):
    supported_arch = {'x86', 'x64', 'arm32', 'arm64', 'mips32', 'mips64'}
    arch_set = set(arch_set)
    assert arch_set.issubset(supported_arch), (f"Meet unsupported architectures: {arch_set - supported_arch}, "
                                               f"supported architectures are: {supported_arch}")


def check_generate_subset_config(config: dict, config_file):
    assert set(config.keys()).issubset(
        {'train', 'validation', 'test'}), 'The configuration file must contain the train, validation and test subsets.'

    for exp_type, value in config['test'].items():
        if isinstance(value, str):
            config_dir = os.path.split(config_file)[0]
            if os.path.isabs(value):
                target_file = value
            else:
                target_file = os.path.join(config_dir, value)
            with open(target_file, 'r') as f:
                config['test'][exp_type] = yaml.load(f, Loader=yaml.FullLoader)

    test_software = set()
    for exp in config['test'].values():
        test_software.update(exp['software'])

    assert len(config['train']) + len(config['validation']) + len(test_software) == len(
        set(config['train']) | set(config['validation']) | test_software), \
        'The software list of train, test and validation subsets must be disjoint.'


def check_merge_config(config, config_file):
    check_and_set_default(f'merge-config', config,
                          default_value={'occurrence-threshold': 0, 'compiler': None},
                          required_kwargs={'remove-duplicate', 'small-graph-threshold',
                                           'large-graph-threshold', 'subset', 'arch'})
    check_generate_subset_config(config['subset'], config_file)
    check_support_arch(config['arch'])
    config['arch'] = set(config['arch'])
    config['compiler'] = set(config['compiler']) if config['compiler'] is not None else None


def check_log_config(config):
    check_and_set_default("log-config", config,
                          default_value={
                              'level': logging.INFO,
                              'console': True
                          })
    init_logger(logger.name, config['level'], console=config['console'], console_level=config['level'])


def check_config(config: dict, config_file):
    check_and_set_default('config-file', config,
                          required_kwargs={'dataset', 'disassemble', 'log', 'merge'}
                          , logger=logger)
    check_dataset_config(config['dataset'])
    check_disassemble_config(config['disassemble'])
    check_merge_config(config['merge'], config_file)
    check_log_config(config['log'])
    # check some extra constraints
    merge_config, disassemble_config = config['merge'], config['disassemble']
    assert merge_config['arch'].issubset(disassemble_config['arch']), ("The architecture set in merge config should be "
                                                                       "subset of the set in disassemble config.")


def get_normalizer(graph_type, normalizer_kwargs, log_config):
    match graph_type:
        case GraphType.PDG:
            from binsim.disassembly.binaryninja.core.graph.pdg import PDGNormalizer, logger
            normalizer_type, normalizer_logger = PDGNormalizer, logger
        case GraphType.TokenCFG:
            from binsim.disassembly.binaryninja.core.graph.TokenCFG import TokenCFGNormalizer, logger
            from binsim.neural.lm import Ins2vec
            normalizer_type, normalizer_logger = TokenCFGNormalizer, logger
            if normalizer_kwargs.get('ins2vec', None) is None:
                logger.warning(f"The graph type is {graph_type}. If you could provide the ins2idx, "
                               f"the normalizer will replace tokens with ids, which can reduce the memory usage.")
                normalizer_kwargs['ins2idx'] = None
            else:
                ins2idx = Ins2vec.load(normalizer_kwargs['ins2vec']).ins2idx
                normalizer_kwargs['ins2idx'] = ins2idx
                del normalizer_kwargs['ins2vec']
                logger.info(f"ins2idx loaded, it contains {len(ins2idx)} vocabulary。")
        case GraphType.ACFG:
            from binsim.disassembly.binaryninja.core.graph.attributeCFG import AttributedCFGNormalizer, logger
            normalizer_type, normalizer_logger = AttributedCFGNormalizer, logger
        case GraphType.ByteCode:
            from binsim.disassembly.binaryninja.core.graph.ByteCode import ByteCodeNormalizer, logger
            normalizer_type, normalizer_logger = ByteCodeNormalizer, logger
        case GraphType.InsCFG:
            from binsim.disassembly.binaryninja.core.graph.InsCFG import InsCFGNormalizer, logger
            normalizer_type, normalizer_logger = InsCFGNormalizer, logger
        case GraphType.MnemonicCFG:
            from binsim.disassembly.binaryninja.core.graph.mnemonicCFG import MnemonicCFGNormalizer, logger
            normalizer_type, normalizer_logger = MnemonicCFGNormalizer, logger
        case GraphType.CodeAST | GraphType.JTransSeq:
            normalizer_type = None
            return normalizer_type, normalizer_kwargs
        case _:
            raise ValueError(f'The graph type({graph_type}) is not supported.')
    init_logger(normalizer_logger.name, level=log_config['level'], console=log_config['console'],
                console_level=log_config['level'])
    return normalizer_type, normalizer_kwargs


def disassemble(config):
    from binsim.disassembly.binaryninja.core import BinaryNinja
    from binsim.disassembly.ida import IDADisassembler
    logger.info(f"The configuration used is:\n {dict2str(config)}")
    dataset_config, disassemble_config = config['dataset'], config['disassemble']
    merge_config, log_config = config['merge'], config['log']
    binaries, dataset = dataset_config['binary'], dataset_config['dataset']
    graph_type, dataset_name = dataset_config['type'], dataset_config['name']
    dataset_dir = DatasetDir(dataset,
                             graph_type=graph_type.value,
                             dataset_name=dataset_name,
                             merge_name=merge_config['name'])
    kwargs = dataset_config['kwargs']
    for key in list(kwargs.keys()):
        if '-' in key:
            kwargs[key.replace('-', '_')] = kwargs.pop(key)
    normalizer_type, normalizer_kwargs = get_normalizer(graph_type, kwargs, log_config=log_config)
    # disassemble each binary file, generate bndb files and cfg files of different types.
    binary_files = list_dataset_binaries(binaries, disassemble_config['arch'], disassemble_config['compiler'])
    target_files = [dataset_dir.rel_to_graph_dir(os.path.relpath(file, binaries)) for file in
                    binary_files]
    logger.info(f"{len(binary_files)} binary files will be disassembled.")

    if graph_type == GraphType.CodeAST or graph_type == GraphType.JTransSeq:
        db_files = [dataset_dir.rel_to_cache_dir(os.path.relpath(file, binaries) + '.i64') for file in binary_files]
        log_files = [dataset_dir.rel_to_log_dir(os.path.relpath(file, binaries) + '.log') for file in binary_files]
        disassembler = IDADisassembler(graph_type, ida_path=disassemble_config['ida_path'])
        disassembler.disassemble_files(
            src_files=binary_files,
            out_files=target_files,
            db_files=db_files,
            log_files=log_files,
            workers=disassemble_config['workers'],
            verbose=disassemble_config['verbose'],
            keep_thunks=disassemble_config['keep-thunks'],
            keep_unnamed=False,
            keep_large=disassemble_config['keep-large'],
            large_ins_threshold=disassemble_config['large-ins-threshold'],
            large_graph_threshold=disassemble_config['large-graph-threshold'],
            keep_small=disassemble_config['keep-small'],
            small_ins_threshold=disassemble_config['small-ins-threshold'],
            small_graph_threshold=disassemble_config['small-graph-threshold'],
            regenerate=disassemble_config['regenerate'],
            reanalysis=disassemble_config['reanalysis'],
            **disassemble_config['kwargs']
        )
    else:
        db_files = [dataset_dir.rel_to_cache_dir(os.path.relpath(file, binaries) + '.bndb') for file in binary_files]
        disassembler = BinaryNinja(normalizer_type,
                                   normalizer_kwargs=kwargs)
        disassembler.disassemble_files(src_files=binary_files,
                                       out_files=target_files,
                                       db_files=db_files,
                                       workers=disassemble_config['workers'],
                                       verbose=disassemble_config['verbose'],
                                       keep_thunks=disassemble_config['keep-thunks'],
                                       keep_unnamed=False,
                                       keep_large=disassemble_config['keep-large'],
                                       large_ins_threshold=disassemble_config['large-ins-threshold'],
                                       large_graph_threshold=disassemble_config['large-graph-threshold'],
                                       keep_small=disassemble_config['keep-small'],
                                       small_ins_threshold=disassemble_config['small-ins-threshold'],
                                       small_graph_threshold=disassemble_config['small-graph-threshold'],
                                       regenerate=disassemble_config['regenerate'],
                                       reanalysis=disassemble_config['reanalysis'])


def remove_same_name_function(graph_db):
    unique_func_name, same_name_function_statistics = set(), defaultdict(lambda: 0)
    for graph_name in list(graph_db):
        software, file, func_name = graph_name.split(':')
        # if current function has same name with another function in same software,
        # it will be removed.
        if (software, func_name) in unique_func_name:
            del graph_db[graph_name]
            same_name_function_statistics[software] += 1
        else:
            unique_func_name.add((software, func_name))
    return same_name_function_statistics


def remove_same_hash_function(graph_db):
    func_hash, hash2func = set(), {}
    same_name_dup_func_count, diff_name_dup_func_count = 0, 0
    diff_name_dup_functions, same_name_dup_functions = set(), set()
    for graph_name, options in list(graph_db.items()):
        cur_graph_hash = set()
        for option, graph in list(options.items()):
            if graph.hash in func_hash:
                del options[option]
                diff_name_dup_func_count += 1
                diff_name_dup_functions.add(graph_name)
                continue
            elif graph.hash in cur_graph_hash:
                del options[option]
                same_name_dup_functions.add(graph_name)
                same_name_dup_func_count += 1
            else:
                cur_graph_hash.add(graph.hash)
        func_hash.update(cur_graph_hash)
        for cur_func_hash in cur_graph_hash:
            hash2func[cur_func_hash] = graph_name
        if len(options) == 0:
            del graph_db[graph_name]
    return ((same_name_dup_func_count, same_name_dup_functions),
            (diff_name_dup_func_count, diff_name_dup_functions))

def load_data(dataset_dir, software_list, merged_arch_set, merged_compiler_set, subset_name, cfg_type,
              small_graph_node_threshold,
              large_graph_node_threshold, occurrence_threshold):
    # iterate different types of graphs
    graph_db, meet_files = {}, set()
    # merge all functions into a dict
    for software in tqdm.tqdm(software_list):
        software_cfg_dir = dataset_dir.rel_to_graph_dir(software)
        versions = sorted(os.listdir(software_cfg_dir))
        versions2idx = {version: str(idx) for idx, version in enumerate(versions)}
        for version in os.listdir(software_cfg_dir):
            version_cfg_dir = os.path.join(software_cfg_dir, version)
            for arch in os.listdir(version_cfg_dir):
                arch_cfg_dir = os.path.join(version_cfg_dir, arch)
                if arch not in merged_arch_set:
                    continue
                for _os in os.listdir(arch_cfg_dir):
                    _os_cfg_dir = os.path.join(arch_cfg_dir, _os)
                    for compiler in os.listdir(_os_cfg_dir):
                        if merged_compiler_set is not None and compiler not in merged_compiler_set:
                            continue
                        compiler_cfg_dir = os.path.join(_os_cfg_dir, compiler)
                        for op_level in os.listdir(compiler_cfg_dir):
                            op_level_cfg_dir = os.path.join(compiler_cfg_dir, op_level)
                            options = (versions2idx[version], arch, _os, compiler, op_level)
                            for file in os.listdir(op_level_cfg_dir):
                                file_path = os.path.join(op_level_cfg_dir, file)

                                # sometimes, the filename contains the version of software, we should try to remove
                                # the version part
                                dot_parts = file.split('.')
                                if not (len(dot_parts) == 2 and (file.endswith('.so') or file.endswith('.o'))
                                        or len(dot_parts) == 1):

                                    quiet = True
                                    if file not in meet_files:
                                        meet_files.add(file)
                                        quiet = False
                                    if not quiet:
                                        logger.warning(f"Meet a file which may contain version string: {file}, try to "
                                                       f"remove the version string.")
                                    name, *suffix = dot_parts
                                    suffix = [name] + [part for part in suffix if part == 'so']
                                    file = '.'.join(suffix)

                                    if not quiet:
                                        logger.warning(f"New name after removing version string is {file}.")

                                with open(file_path, 'rb') as f:
                                    graph_list = pickle.load(f)
                                for graph in graph_list:
                                    graph_name = f'{software}:{file}:{graph.name}'
                                    if graph_name not in graph_db:
                                        graph_db[graph_name] = dict()
                                    graph_db[graph_name][options] = graph

    # 1. statistic number of functions
    # 2. remove duplicated functions
    #    1) functions with same name, but in different files;
    #    2) functions with same hash, but has different name.

    same_name_function_statistics = remove_same_name_function(graph_db)
    total_removed = sum(same_name_function_statistics.values())
    logger.info(f"{total_removed:,} same name functions are removed from the {subset_name} set.")
    logger.info(f"Concrete statistics: {dict2str(same_name_function_statistics)}.")

    # remove duplicate functions for train and validation
    # if two functions have same hash, but have different compilation options or different names,
    # one of them can be removed. This phenomenon usually may occur when
    #   1. two function with different names have same source code.
    #   2. optimizations doesn't change the binary code of functions.
    # For test, this operation will be done during tests.
    # (same_name_dup_func_count, same_name_dup_functions), (diff_name_dup_func_count, diff_name_dup_functions) = \
    #     remove_same_hash_function(graph_db)
    # dup_func_count = same_name_dup_func_count + diff_name_dup_func_count
    # logger.info(f"{dup_func_count:,} duplicated graphs are removed from the {subset_name} set.")
    # logger.info(
    #     f"{same_name_dup_func_count:,} duplicated graphs ({len(same_name_dup_functions):,} unique functions) "
    #     f"are removed, as their code are same under different compilation options.")
    # logger.info(
    #     f"{diff_name_dup_func_count:,} duplicated graphs ({len(diff_name_dup_functions):,} unique functions)"
    #     f" with different names are removed, the reason should be further inspected.")

    # remove functions according to node_threshold
    for graph_name in list(graph_db.keys()):
        for option in list(graph_db[graph_name].keys()):
            cfg = graph_db[graph_name][option]
            if cfg.node_num < small_graph_node_threshold or cfg.node_num > large_graph_node_threshold:
                del graph_db[graph_name][option]
        if len(graph_db[graph_name]) < occurrence_threshold:
            del graph_db[graph_name]

    # statistic function distribution
    software_function_statistics, function_individual_statistics = defaultdict(int), defaultdict(int)
    for graph_name, options in graph_db.items():
        software_name, _, function_name = graph_name.split(":")
        software_function_statistics[software_name] += 1
        function_individual_statistics[len(options)] += 1

    for software, unique_software_func_num in software_function_statistics.items():
        logger.info(
            f"{unique_software_func_num:,} unique functions are extracted from {software} in the {subset_name}-set.")

    unique_func_num, total_graph_num = len(graph_db), sum(map(len, graph_db.values()))
    logger.info(
        f"{total_graph_num:,} graphs ({unique_func_num:,} unique functions) are extracted from the {subset_name}-set.")

    with open(dataset_dir.rel_to_data_dir(f'{subset_name}-statistics.pkl'), 'wb') as out:
        pickle.dump({
            'total_graph_num': total_graph_num,
            'unique_func_num': unique_func_num,
            # 'same_name_dup_func_count': same_name_dup_func_count,
            # 'diff_name_dup_functions': diff_name_dup_functions,
            'function_individual_statistics': function_individual_statistics
        }, out)

    if cfg_type == GraphType.MnemonicCFG:
        logger.info(
            "As current graph type is MnemonicCFG, we need to calculate the ins2idx and replace mnemonics with ids.")
        min_freq = 10
        if subset_name == 'train':
            ins2idx = generate_ins2idx_for_mnemonic_cfg(graph_db, min_freq)
            with open(dataset_dir.rel_to_data_dir("token2idx.pkl"), 'wb') as f:
                pickle.dump(ins2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(dataset_dir.rel_to_data_dir("token2idx.pkl"), 'rb') as f:
                ins2idx = pickle.load(f)
        for graph_name, graphs in graph_db.items():
            for graph in graphs.values():
                graph.replace_tokens(ins2idx)
    elif cfg_type in [GraphType.InsCFG, GraphType.TokenCFG]:
        if subset_name == 'train':
            logger.info(f"As current graph type is {cfg_type}, "
                        f"we need to calculate the ins2idx and replace ins with ids.")
            tokens2idx = {'<PAD>': 1, '<UNK>': 0}
            for graph_name, graphs in graph_db.items():
                for graph in graphs.values():
                    graph.replace_tokens(tokens2idx, record_unseen=False, update_token2id=True)
            logger.info(f"ins2idx generated. There are {len(tokens2idx)} tokens in the vocabulary.")
            # save ins2idx
            with open(dataset_dir.rel_to_data_dir("token2idx.pkl"), 'wb') as f:
                pickle.dump(dict(tokens2idx), f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(dataset_dir.rel_to_data_dir("token2idx.pkl"), 'rb') as f:
                tokens2idx = pickle.load(f)
            tokens2idx = tokens2idx
            unseen_tokens = set()
            total_ins_num, total_unseen_ins_num = 0, 0
            for graph_name, graphs in graph_db.items():
                for graph in graphs.values():
                    cur_unseen_tokens, cur_ins_num, cur_unseen_token_num = (
                        graph.replace_tokens(tokens2idx, record_unseen=True, update_token2id=False))
                    unseen_tokens.update(cur_unseen_tokens)
                    total_ins_num += cur_ins_num
                    total_unseen_ins_num += cur_unseen_token_num
            with open(dataset_dir.rel_to_data_dir(f"{subset_name}-unseen-tokens.txt"), 'wb') as f:
                pickle.dump((unseen_tokens, total_ins_num, total_unseen_ins_num),
                            f, protocol=pickle.HIGHEST_PROTOCOL)

    return graph_db

def generate_ins2idx_for_mnemonic_cfg(graph_db, min_freq):
    counter = Counter()
    for graph_name, graphs in graph_db.items():
        for graph in graphs.values():
            counter.update(graph.unique_tokens())
    ins2idx = {'<PAD>': 1, '<UNK>': 0}
    for token, freq in counter.items():
        if freq >= min_freq:
            ins2idx[token] = len(ins2idx)
    logger.info(f"ins2idx generated. There are {len(ins2idx):,} tokens in the vocabulary.")
    logger.info(f"The most frequent tokens are {counter.most_common(10)}.")
    return ins2idx

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


def filter_func_with_comb(comb):
    def filter_func(name, functions):
        if comb is not None:
            return {comb: functions[comb]} if comb in functions else dict()
        else:
            return functions

    return filter_func

def merge_dataset(config):
    dataset_config, merge_config, log_config = config['dataset'], config['merge'], config['log']

    # check whether the dataset exists.
    dataset, cfg_type, dataset_name = dataset_config['dataset'], dataset_config['type'], dataset_config['name']
    remove_duplicate, merge_name = merge_config['remove-duplicate'], merge_config['name']
    subsets = merge_config['subset']
    dataset_dir = DatasetDir(dataset, cfg_type.value, dataset_name=dataset_name, merge_name=merge_name)
    merged_arch_set = merge_config['arch']
    merged_compiler_set = merge_config['compiler']
    # os.path.join(dataset, GRAPH_DIR, cfg_type, dataset_name)
    # all software should have been disassembled.
    software_list = subsets['train'] + subsets['validation']
    for exp_type in subsets['test']:
        software_list += subsets['test'][exp_type]['software']

    for software_name in software_list:
        assert os.path.exists(dataset_dir.rel_to_graph_dir(software_name)), \
            f'The {cfg_type} has not been generated for {software_name} in the dataset({dataset}).'

    # merge train-set and validation set
    for subset_name in ['train', 'validation']:
    # for subset_name in ['train', 'validation']:
        software_list = subsets[subset_name]
        logger.info(f"Generating {subset_name}-set.")
        logger.info(f"{len(software_list):,} software({','.join(software_list)}) will be merged.")
        data_dir = dataset_dir.data_dir
        graph_db = load_data(dataset_dir, software_list, merged_arch_set, merged_compiler_set, subset_name, cfg_type,
                             small_graph_node_threshold=merge_config['small-graph-threshold'],
                             large_graph_node_threshold=merge_config['large-graph-threshold'],
                             occurrence_threshold=merge_config['occurrence-threshold'],
                             )
        with open(os.path.join(data_dir, subset_name + '.pkl'), 'wb') as f:
            pickle.dump(graph_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    subset_name = 'test'
    for exp_type, exp_config in subsets['test'].items():
        pool_size = exp_config['pool-size']
        pool_number = exp_config['number']
        seed = exp_config['seed']
        software_list = exp_config['software']
        # 1. load all functions according to software_list
        graph_db = load_data(dataset_dir, software_list, merged_arch_set,  merged_compiler_set, 'test', cfg_type,
                             small_graph_node_threshold=merge_config['small-graph-threshold'],
                             large_graph_node_threshold=merge_config['large-graph-threshold'],
                             occurrence_threshold=merge_config['occurrence-threshold'])
        # 2. generate test pools for each option combination
        random.seed(seed)
        for option_name, (option1, option2) in tqdm.tqdm(exp_config['options'].items()):
            # 2.1 find out functions compiled under option1 and option2
            option1, option2 = tuple(option1.split(':')), tuple(option2.split(':'))
            samples1, names1, tags1 = extract_samples_from_data(graph_db,
                                                                filter_func=filter_func_with_comb(option1))
            samples2, names2, tags2 = extract_samples_from_data(graph_db,
                                                                filter_func=filter_func_with_comb(option2))

            common_names = set(names2) & set(names1)
            assert len(common_names) >= pool_size, (f"In {option1} vs {option2}, function numbers({len(common_names)}) "
                                                    f"is too small, less than pool-size({pool_size}).")
            functions1 = [(sample, name, tag) for sample, name, tag in zip(samples1, names1, tags1)
                                 if name in common_names]
            functions1.sort(key=lambda x: x[1])

            functions2 = [(sample, name, tag) for sample, name, tag in zip(samples2, names2, tags2)
                                 if name in common_names]
            functions2.sort(key=lambda x: x[1])

            # 2.2 generate pools
            os.makedirs(dataset_dir.rel_to_data_dir(f"{subset_name}/{exp_type}/{option_name}/"), exist_ok=True)
            for pool_idx in range(pool_number):
                sample_index = list(range(len(functions1)))
                random.shuffle(sample_index)
                selected_sample_index = sample_index[:pool_size]
                selected_function1 = [functions1[idx] for idx in selected_sample_index]
                selected_function2 = [functions2[idx] for idx in selected_sample_index]
                with open(dataset_dir.rel_to_data_dir(f"{subset_name}/{exp_type}/{option_name}/{pool_idx}.pkl"), "wb") as f:
                    pickle.dump((selected_function1, selected_function2), f)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess binaries and generate datasets.')
    parser.add_argument('--config', type=str, required=True, help='The configuration file which provides .')
    sub_parser = parser.add_subparsers(dest='action', required=True)
    sub_parser.add_parser("disassemble",
                          help="Disassemble each binary file, generate bndb files and cfg files of different types.")
    sub_parser.add_parser("merge",
                          help="Merge generated cfg files and generate subsets for training, validation and testing.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_file = os.path.abspath(args.config)
    assert os.path.exists(config_file), f'The configuration file({config_file}) does not exist.'
    logger.info(f"Load configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    check_config(config, config_file)
    if args.action == 'disassemble':
        disassemble(config)
    elif args.action == 'merge':
        merge_dataset(config)


if __name__ == '__main__':
    main()
