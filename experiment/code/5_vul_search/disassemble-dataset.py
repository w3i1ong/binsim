import os
import yaml
import logging
import argparse
from typing import List, Union
from binsim.disassembly.utils.globals import GraphType
from binsim.disassembly.utils import get_architecture
from binsim.utils import dict2str, init_logger, check_and_set_default
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
                                # skip no-executables
                                try:
                                    if get_architecture(file_path) != _arch:
                                        continue
                                except ValueError:
                                    continue
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

def check_log_config(config):
    check_and_set_default("log-config", config,
                          default_value={
                              'level': logging.INFO,
                              'console': True
                          })
    init_logger(logger.name, config['level'], console=config['console'], console_level=config['level'])


def check_config(config: dict, config_file):
    check_and_set_default('config-file', config,
                          required_kwargs={'dataset', 'disassemble', 'log'}
                          , logger=logger)
    check_dataset_config(config['dataset'])
    check_disassemble_config(config['disassemble'])
    check_log_config(config['log'])


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
        case GraphType.InsCFG:
            from binsim.disassembly.binaryninja.core.graph.InsCFG import InsCFGNormalizer, logger
            normalizer_type, normalizer_logger = InsCFGNormalizer, logger
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
    log_config = config['log']
    binaries, dataset = dataset_config['binary'], dataset_config['dataset']
    graph_type, dataset_name = dataset_config['type'], dataset_config['name']
    dataset_dir = DatasetDir(dataset,
                             graph_type=graph_type.value,
                             dataset_name=dataset_name,
                             merge_name="default")
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
                                   normalizer_kwargs=normalizer_kwargs)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess binaries and generate datasets.')
    parser.add_argument('--config', type=str, required=True, help='The configuration file which provides .')
    return parser.parse_args()


def main():
    args = parse_args()
    config_file = os.path.abspath(args.config)
    assert os.path.exists(config_file), f'The configuration file({config_file}) does not exist.'
    logger.info(f"Load configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    check_config(config, config_file)
    disassemble(config)


if __name__ == '__main__':
    main()
