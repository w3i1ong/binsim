import os
import argparse
from binsim.disassembly.utils import get_architecture, is_elf_file
from binsim.utils import init_logger
from hashlib import sha256
from collections import defaultdict

logger = init_logger("Check-Dataset", level="INFO", console=True, console_level="INFO")

def get_first_4k_hash(filename):
    with open(filename, 'rb') as f:
        return sha256(f.read(4096)).hexdigest()

def get_file_hash(filename):
    with open(filename, 'rb') as f:
        return sha256(f.read()).hexdigest()

def parse_args():
    parser = argparse.ArgumentParser(description='Check the validity of dataset.')
    parser.add_argument('--dataset-dir', type=str, required=True, help='The directory of the dataset'
                                                                      '(should be the converted).')
    return parser.parse_args()

def check_dataset(dataset_dir):
    logger.info(f"Start checking the dataset at {dataset_dir}.")
    check_file_cnt = 0
    hash2file = defaultdict(set)
    duplicate_files = defaultdict(set)
    for software in os.listdir(dataset_dir):
        software_dir = os.path.join(dataset_dir, software)
        for version in os.listdir(software_dir):
            version_dir = os.path.join(software_dir, version)
            for arch in os.listdir(version_dir):
                arch_dir = os.path.join(version_dir, arch)
                for os_name in os.listdir(arch_dir):
                    os_dir = os.path.join(arch_dir, os_name)
                    for compiler in os.listdir(os_dir):
                        compiler_dir = os.path.join(os_dir, compiler)
                        for opt_level in os.listdir(compiler_dir):
                            opt_level_dir = os.path.join(compiler_dir, opt_level)
                            for file in os.listdir(opt_level_dir):
                                file_path = os.path.join(opt_level_dir, file)
                                assert os.path.isfile(file_path), f"{file_path} is not a file."
                                # sometimes, the file may not be an ELF file, just skip it
                                if not is_elf_file(file_path):
                                    logger.warning(f"Meet a non-ELF file: {file_path}.")
                                    continue
                                check_file_cnt += 1
                                # check whether the architecture of the file is same as its directory name
                                assert get_architecture(file_path) == arch, \
                                    (f"The architecture of {file_path} is invalid."
                                     f"(Expected: {arch}, Actual: {get_architecture(file_path)})")
                                # check whether the hash of the first 1k bytes of the file is unique
                                file_hash = get_file_hash(file_path)
                                if file_hash in hash2file:
                                    duplicate_files[file_hash].add(file_path)
                                hash2file[file_hash].add(file_path)
    if len(duplicate_files) > 0:
        logger.error(f"Meet {len(duplicate_files)} duplicate files.")
    logger.info(f"{check_file_cnt} files have been checked. The dataset is valid.")

def main():
    args = parse_args()
    check_dataset(args.dataset_dir)

if __name__ == '__main__':
    main()