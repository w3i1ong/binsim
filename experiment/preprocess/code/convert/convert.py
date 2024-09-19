import os
import shutil
import argparse
from enum import Enum
from tqdm import tqdm

class Dataset(Enum):
    MINE = 'mine'
    BINARYCORP = 'binarycorp'
    CISCO = 'cisco'
    TREX = 'trex'
    BINKIT = 'binkit'

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--original', type=str, required=True, help='The directory of the original dataset.')
    parser.add_argument('--converted', type=str, required=True, help='The directory of the converted dataset.')
    sub_parsers = parser.add_subparsers(dest='Type', required=True)
    sub_parsers.add_parser(Dataset.CISCO.value)
    sub_parsers.add_parser(Dataset.MINE.value)
    sub_parsers.add_parser(Dataset.BINARYCORP.value)
    sub_parsers.add_parser(Dataset.TREX.value)
    sub_parsers.add_parser(Dataset.BINKIT.value)
    return parser.parse_args()


def convert_cisco_dataset_1(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for software in tqdm(os.listdir(src_dir)):
        software_dir = os.path.join(src_dir, software)
        for file in os.listdir(software_dir):
            options, binary = file.split('_')
            arch, compiler, compiler_version, opt_level = options.split('-')

            dst_file_dir = os.path.join(dst_dir, software, 'ukn', arch, 'linux', f'{compiler}-{compiler_version}',
                                        opt_level)
            os.makedirs(dst_file_dir, exist_ok=True)

            dst_file_path = os.path.join(dst_file_dir, binary)
            src_file_path = os.path.join(software_dir, file)
            shutil.copy(src_file_path, dst_file_path)


def convert_cisco_dataset_2(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for file in tqdm(os.listdir(src_dir)):
        if os.path.isdir(os.path.join(src_dir, file)):
            continue
        arch, software, binary = file.split('_')
        if arch == 'x86-32':
            arch = 'x86'
        elif arch == 'x86-64':
            arch = 'x64'
        else:
            arch = arch.replace('-', '')
        software, version_opt_level = software.split('-', 1)

        # deal with some special cases, like 'ImageMagick-7.0.10-27-O2'
        *version, opt_level = version_opt_level.split('-')
        version = '-'.join(version)

        dst_file_dir = os.path.join(dst_dir, software, version, arch, 'linux', 'gcc', opt_level)
        os.makedirs(dst_file_dir, exist_ok=True)

        dst_file_path = os.path.join(dst_file_dir, binary)
        src_file_path = os.path.join(src_dir, file)
        shutil.copy(src_file_path, dst_file_path)


def convert_cisco_dataset(src_dir, dst_dir):
    dataset_1_src_dir = os.path.join(src_dir, 'Dataset-1')
    dataset_1_dst_dir = os.path.join(dst_dir, 'Dataset-1')
    dataset_2_src_dir = os.path.join(src_dir, 'Dataset-2')
    dataset_2_dst_dir = os.path.join(dst_dir, 'Dataset-2')
    convert_cisco_dataset_1(dataset_1_src_dir, dataset_1_dst_dir)
    convert_cisco_dataset_2(dataset_2_src_dir, dataset_2_dst_dir)


def convert_binarycorp_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for subset in ['train', 'test', 'small_train']:
        subset_dir = os.path.join(src_dir, subset)
        for file in tqdm(os.listdir(subset_dir)):
            if os.path.isdir(os.path.join(subset_dir, file)):
                continue
            *filename, opt_level, _ = file.split('-')
            filename = '-'.join(filename)
            dst_file_dir = os.path.join(dst_dir, subset, 'ukn', 'x64', 'linux', 'gcc', opt_level)
            os.makedirs(dst_file_dir, exist_ok=True)

            dst_file_path = os.path.join(dst_file_dir, filename)
            src_file_path = os.path.join(subset_dir, file)
            shutil.copy(src_file_path, dst_file_path)


def convert_mine_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for software in os.listdir(src_dir):
        software_dir = os.path.join(src_dir, software)
        for options in os.listdir(software_dir):
            compiler, compiler_version, arch, _, _, opt_level, *version = options.split('-')
            version = '-'.join(version)

            if arch == 'aarch64':
                arch = 'arm64'

            dst_file_dir = os.path.join(dst_dir, software, version, arch, 'linux', f'{compiler}-{compiler_version}',
                                        f'O{opt_level}')
            os.makedirs(dst_file_dir, exist_ok=True)
            src_file_dir = os.path.join(software_dir, options)

            for file in os.listdir(src_file_dir):
                dst_file_path = os.path.join(dst_file_dir, file)
                src_file_path = os.path.join(src_file_dir, file)
                shutil.copy(src_file_path, dst_file_path)


def convert_trex_dataset(src_dir, dst_dir):
    supported_arch_list = ['x86-32', 'x86-64', 'arm-32', 'mips-32']
    normal_arch = {'x86-32': 'x86', 'x86-64': 'x64', 'arm-32': 'arm32', 'mips-32': 'mips32'}
    os.makedirs(dst_dir, exist_ok=True)
    software_set = None
    for arch in os.listdir(src_dir):
        # only convert x86-32, x86-64, arm32, mips32
        if arch not in supported_arch_list:
            continue
        current_software_set = set()
        arch_dir = os.path.join(src_dir, arch)
        for software in os.listdir(arch_dir):
            name, version_op = software.split('-', maxsplit=1)
            version, opt_level = version_op.rsplit('-', maxsplit=1)
            # skip obfuscated binaries
            if not opt_level.startswith('O'):
                continue
            current_software_set.add(f'{name}-{version}')
        if software_set is None:
            software_set = current_software_set
        else:
            software_set = software_set.intersection(current_software_set)
    # remove software copies of different version
    unique_software_set = set([software.split('-')[0] for software in software_set])
    for software in list(software_set):
        name = software.split('-')[0]
        if name not in unique_software_set:
            software_set.remove(software)
        else:
            unique_software_set.remove(name)
    for arch in supported_arch_list:
        arch_dir = os.path.join(src_dir, arch)
        arch = normal_arch[arch]
        for software in os.listdir(arch_dir):
            name, version_op = software.split("-", maxsplit=1)
            version, opt_level = version_op.rsplit('-', maxsplit=1)
            if f'{name}-{version}' not in software_set:
                continue
            if not opt_level.startswith('O'):
                continue
            compiler, compiler_version = 'ukn', 'ukn'
            dst_file_dir = os.path.join(dst_dir, name, version, arch, 'linux', f'{compiler}-{compiler_version}',opt_level)
            os.makedirs(dst_file_dir, exist_ok=True)
            src_file_dir = os.path.join(arch_dir, software)
            for file in os.listdir(src_file_dir):
                dst_file_path = os.path.join(dst_file_dir, file)
                src_file_path = os.path.join(src_file_dir, file)
                if dst_file_path.endswith(".a"):
                    os.system(f"ar -x {src_file_path} --output={dst_file_dir}")
                else:
                    shutil.copy(src_file_path, dst_file_path)

def convert_binkit_dataset(src_dir, dst_dir):
    for software in os.listdir(src_dir):
        software_dir, dst_software_dir = os.path.join(src_dir, software), os.path.join(dst_dir, software)
        for file in os.listdir(software_dir):
            src_file_path = os.path.join(software_dir, file)
            name_version, compiler_version, arch_optim = file.split('_', maxsplit=2)
            software_name, software_version = name_version.split('-')
            compiler, compiler_version = compiler_version.split('-')
            arch, bits, opt_level, filename = arch_optim.split('_', maxsplit=3)
            dst_file_dir = os.path.join(dst_software_dir, software_version, arch, 'linux',
                                        f'{compiler}-{compiler_version}', opt_level)
            os.makedirs(dst_file_dir, exist_ok=True)
            dst_file_path = os.path.join(dst_file_dir, filename)
            shutil.copy(src_file_path, dst_file_path)

def main():
    args = parse_args()

    match Dataset(args.Type):
        case Dataset.CISCO:
            convert_cisco_dataset(args.original, args.converted)
        case Dataset.MINE.value:
            convert_mine_dataset(args.original, args.converted)
        case Dataset.BINARYCORP:
            convert_binarycorp_dataset(args.original, args.converted)
        case Dataset.TREX:
            convert_trex_dataset(args.original, args.converted)
        case Dataset.BINKIT:
            convert_binkit_dataset(args.original, args.converted)


if __name__ == '__main__':
    main()
