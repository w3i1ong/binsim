import os, glob, yaml, pickle, logging, argparse
from typing import List, Tuple
from binsim_binary.archinfo import Arch
from binsim.disassembly.utils.globals import GraphType
from binsim.utils import dict2str, init_logger, check_and_set_default
from collections import defaultdict
from binsim.disassembly.extractor import get_extractor_by_name

logger = init_logger(os.path.basename(__file__), logging.INFO,
                     console=True, console_level=logging.INFO)

def transform_kwargs(kwargs:dict):
    for key in list(kwargs):
        if '-' in key:
            new_key = key.replace('-', '_')
            kwargs[new_key] = kwargs.pop(key)
    return kwargs

def list_dataset_binaries(dataset: str, software_list: List[str], arch_to_deal: List[Arch], compilers_to_deal: List[str])\
        -> Tuple[List[str], List[str], List]:
    binaries, file_project, file_options = [], [], []
    for file_path in glob.glob(f"{dataset}/*/*/*/*/*/*/*"):
        rel_path = os.path.relpath(file_path, dataset)
        software, version, arch, platform, compiler, op_level, file = rel_path.split('/')
        arch = Arch.from_string(arch)
        if arch not in arch_to_deal or compiler not in compilers_to_deal:
            continue
        if software not in software_list:
            continue
        if os.path.islink(file_path):
            continue
        if file.endswith('.symbols') or file.endswith(".a"):
            continue
        binaries.append(file_path)
        file_project.append(software)
        file_options.append((arch.value, platform, compiler,op_level))
    return binaries, file_project, file_options


def check_dataset_config(config: dict):
    check_and_set_default('dataset-config', config,
                          default_value={"remove-duplicate": True, "occurrence-threshold": 0},
                          required_kwargs={'type', 'binary-dir', 'dataset-dir', "cache-dir", "middle-dir",
                                           "arch", "subsets", "compiler"},
                          logger=logger)
    assert os.path.exists(config['binary-dir']), f'The directory of original binaries({config["binary"]}) does not exist.'
    config['type'] = GraphType(config['type'])
    config['arch'] = [Arch.from_string(arch) for arch in config['arch']]
    check_generate_subset_config(config['subsets'])


def check_extractor_config(config: dict):
    check_and_set_default("disassemble-config",
                          kwargs=config,
                          default_value={
                              "workers": 20, "verbose": False, "keep-thunk": False, "keep-large": False,
                              "large-ins-threshold": 3000, "large-graph-threshold": 300,
                              "keep-small": False, "small-ins-threshold": 10, "small-graph-threshold": 5,
                              "regenerate": False, "debug": False, "neural-input-kwargs": {}, "incremental": False,
                              "checkpoint": True
                          },
                          logger=logger)
    assert config['workers'] >= 0, f'The number of workers should not be greater than 0, but got {config["workers"]}.'
    transform_kwargs(config['neural-input-kwargs'])
def check_disassemble_config(config: dict):
    check_and_set_default("disassemble-config",
                          kwargs=config,
                          default_value={"normalizer-kwargs" : dict(), "disassemble-kwargs" : dict(), "neural-input-kwargs": dict()},
                          required_kwargs={"extractor", "normalizer-kwargs"},
                          logger=logger)
    check_extractor_config(config["extractor"])
    transform_kwargs(config["extractor"])
    transform_kwargs(config["normalizer-kwargs"])
    transform_kwargs(config["disassemble-kwargs"])
    transform_kwargs(config["neural-input-kwargs"])


def check_generate_subset_config(config: dict):
    assert set(config.keys()).issubset(
        {'train', 'validation', 'test'}), 'The configuration file must contain the train, validation and test subsets.'
    software_num, software_set = 0, set()
    for subset in config.keys():
        software_num += len(config.get(subset, []))
        software_set.update(config.get(subset, []))
    assert software_num == len(software_set), 'The software list of train, test and validation subsets must be disjoint.'


def check_log_config(config):
    check_and_set_default("log-config", config,
                          default_value={
                              'level': logging.INFO,
                              'console': True
                          })
    init_logger(logger.name, config['level'], console=config['console'], console_level=config['level'])

def load_and_check_config(config_file):
    assert os.path.exists(config_file), f'The configuration file({config_file}) does not exist.'
    logger.info(f"Load configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    check_and_set_default('config-file', config,
                          required_kwargs={'dataset', 'disassemble', 'log'}, logger=logger)
    check_dataset_config(config['dataset'])
    check_disassemble_config(config['disassemble'])
    check_log_config(config['log'])
    return config

def load_extractor_by_name(graph_type, dataset, subset, extractor_kwargs, disassemble_kwargs, normalizer_kwargs, neural_input_kwargs):
    if graph_type in [GraphType.TokenCFG, GraphType.TokenSeq, GraphType.InsCFG, GraphType.InsSeq]:
        if extractor_kwargs.get("token2id_file", None) is None:
            extractor_kwargs["token2id_file"] = os.path.join(dataset, "train", "token2id.pkl")
            extractor_kwargs["record_token2id"] = (subset == "train")
        else:
            extractor_kwargs["record_token2id"] = False
    if graph_type in [GraphType.TokenCFG, GraphType.TokenSeq]:
        if subset == "train":
            extractor_kwargs["corpus_file"] = os.path.join(dataset, "train", "corpus.txt")
        else:
            extractor_kwargs["corpus_file"] = None
    return get_extractor_by_name(graph_type, extractor_kwargs, disassemble_kwargs=disassemble_kwargs,
                                 normalizer_kwargs=normalizer_kwargs, neural_input_kwargs=neural_input_kwargs)

def save_extractor_data(graph_type, extractor, dataset, subset):
    match subset:
        case "train":
            if graph_type in [GraphType.TokenCFG, GraphType.TokenSeq, GraphType.InsCFG]:
                with open(os.path.join(dataset, "train", "token2id.pkl"), "rb") as f:
                    token2id = pickle.load(f)
                id2token = {v: k for k, v in token2id.items()}
                with open(os.path.join(dataset, "train", "id2token.pkl"), "wb") as f:
                    pickle.dump(id2token, f, protocol=pickle.HIGHEST_PROTOCOL)

def disassemble(config):
    logger.info(f"The configuration used is:\n {dict2str(config)}")
    dataset_config, disassemble_config = config['dataset'], config['disassemble']
    binaries, dataset, middle_dir, cache_dir = dataset_config['binary-dir'], dataset_config['dataset-dir'], \
                                                dataset_config['middle-dir'], dataset_config['cache-dir']
    graph_type, log_config = dataset_config['type'], config['log']
    os.makedirs(dataset, exist_ok=True)

    normalizer_kwargs, extractor_kwargs = disassemble_config['normalizer-kwargs'], disassemble_config['extractor']
    disassemble_kwargs, neural_input_kwargs = disassemble_config['disassemble-kwargs'], disassemble_config['neural-input-kwargs']

    subsets = dataset_config["subsets"]
    # Don't modify this! Because the order of these subsets is important, some extractor may generate some information
    # (like token2idx dict) from the training set and use it in the validation set and test set.
    for subset in ["train", "validation", "test"]:
        if subset not in subsets:
            continue
        extractor = load_extractor_by_name(graph_type, dataset, subset, extractor_kwargs=extractor_kwargs,
                                           normalizer_kwargs=normalizer_kwargs, disassemble_kwargs=disassemble_kwargs,
                                           neural_input_kwargs=neural_input_kwargs)
        os.makedirs(os.path.join(dataset, subset), exist_ok=True)
        # disassemble each binary file, generate bndb files and cfg files of different types.
        binary_files, file_project, file_options = (
            list_dataset_binaries(binaries,subsets[subset],
                                  dataset_config['arch'],
                                  dataset_config['compiler']))
        # generate temporary directory to store the intermediate files.
        tmp_dir = os.path.join(middle_dir, subset)
        target_files = [os.path.join(tmp_dir, os.path.relpath(file, binaries)) for file in
                        binary_files]

        logger.info(f"{len(binary_files)} binary files will be disassembled.")
        if graph_type == GraphType.CodeAST or graph_type == GraphType.JTransSeq:
            db_files = [os.path.join(cache_dir, os.path.relpath(file, binaries)) for file in binary_files]
        else:
            db_files = [os.path.join(cache_dir, os.path.relpath(file, binaries) + ".bndb") for file in binary_files]

        dataset_dir = os.path.join(dataset, subset, "dataset.db")
        extractor.disassemble_files(src_files=binary_files,
                                    out_files=target_files,
                                    db_files=db_files,
                                    dataset_dir=dataset_dir)
        # merge target_files
        graph_db = merge_data(dataset, target_files, file_project, file_options, subset,
                              occurrence_threshold=dataset_config["occurrence-threshold"],
                              remove_duplicate=dataset_config["remove-duplicate"])
        with open(os.path.join(dataset, subset, "meta.pkl"), "wb") as f:
            pickle.dump(graph_db, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_extractor_data(graph_type, extractor, dataset, subset)

def remove_same_name_function(graph_db):
    unique_func_name, same_name_function_statistics = set(), defaultdict(lambda: 0)
    same_name_graph_statistics = defaultdict(lambda: 0)
    for graph_name in list(graph_db):
        software, file, func_name = graph_name.split(':')
        # if current function has same name with another function in same software,
        # it will be removed.
        if (software, func_name) in unique_func_name:
            same_name_graph_statistics[software] += len(graph_db[graph_name])
            same_name_function_statistics[software] += 1
            del graph_db[graph_name]
        else:
            unique_func_name.add((software, func_name))
    return same_name_function_statistics, same_name_graph_statistics

def merge_data(dataset_dir, target_files:List[str], file_project, file_options, subset_name,
               occurrence_threshold=True, remove_duplicate=True):
    # iterate different types of graphs
    graph_db, meet_files = {}, set()
    # merge all functions into a dict
    for filepath, software, options in zip(target_files, file_project, file_options):
        if not os.path.exists(filepath):
            continue
        basename = os.path.basename(filepath)
        # sometimes, the filename contains the version of software, we should try to remove
        # the version part
        dot_parts = basename.split('.')
        if not (len(dot_parts) == 2 and (basename.endswith('.so') or basename.endswith('.o'))
                or len(dot_parts) == 1):

            quiet = True
            if basename not in meet_files:
                meet_files.add(basename)
                quiet = False
            if not quiet:
                logger.warning(f"Meet a file which may contain version string: {basename}, try to "
                               f"remove the version string.")
            name, *suffix = dot_parts
            suffix = [name] + [part for part in suffix if part == 'so']
            file = '.'.join(suffix)

            if not quiet:
                logger.warning(f"New name after removing version string is {file}.")

        with open(filepath, 'rb') as f:
            graph_list = pickle.load(f)
        for graph in graph_list:
            graph_name = f'{software}:{basename}:{graph.name}'
            if graph_name not in graph_db:
                graph_db[graph_name] = dict()
            graph_db[graph_name][options] = graph

    if remove_duplicate:
        # 1. statistic number of functions
        # 2. remove duplicated functions（functions with same name, but in different files)
        same_name_function_statistics, same_name_graph_statistics = remove_same_name_function(graph_db)
        total_function_removed = sum(same_name_function_statistics.values())
        total_graph_removed = sum(same_name_graph_statistics.values())
        logger.info(f"{total_function_removed:,} same name functions ({total_graph_removed:,} unique graphs) are removed "
                    f"from the {subset_name} set.")
        logger.info(f"Concrete statistics: {dict2str(same_name_function_statistics)}.")
        logger.info(f"Concrete statistics: {dict2str(same_name_graph_statistics)}.")

    if occurrence_threshold > 0:
        # remove functions according to occurrence_threshold
        for graph_name in list(graph_db.keys()):
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

    with open(os.path.join(dataset_dir, subset_name, f'statistics.pkl'), 'wb') as out:
        pickle.dump({
            'total_graph_num': total_graph_num,
            'unique_func_num': unique_func_num,
            'function_individual_statistics': function_individual_statistics
        }, out)
    return graph_db

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess binaries and generate datasets.')
    parser.add_argument('--config', type=str, required=True, help='The configuration file which provides .')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_and_check_config(args.config)
    disassemble(config)

if __name__ == '__main__':
    main()
