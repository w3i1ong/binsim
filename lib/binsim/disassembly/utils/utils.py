import os


def is_elf_file(filename):
    with open(filename, 'rb') as f:
        data = f.read(16)
        return data.startswith(b'\x7fELF')

def get_architecture(filename):
    with open(filename, 'rb') as f:
        data = f.read(16)
        if not data.startswith(b'\x7fELF'):
            raise ValueError("The provided file is not an ELF file.")
        if data[4] == 1:
            data = f.read(4)[2:]
            if data == b'\x03\x00':
                return 'x86'
            elif data == b'\x28\x00':
                return 'arm32'
            elif data == b'\x00\x08':
                return 'mips32'
            raise ValueError(f"Cannot recognize the architecture of the provided file, whose e_machine is {data.hex()}")
        elif data[4] == 2:
            data = f.read(4)[2:]
            if data == b'\x3e\x00':
                return 'x64'
            elif data == b'\xb7\x00':
                return 'arm64'
            elif data == b'\x00\x08':
                return 'mips64'
            raise ValueError(f"Cannot recognize the architecture of the provided file, whose e_machine is {data.hex()}")


def strip_file_local(infile, outfile):
    arch = get_architecture(infile)
    match arch:
        case 'x86':
            ret_code = os.system(f'strip --strip-all {infile} -o {outfile} > /dev/null 2>&1')
        case 'x64':
            ret_code = os.system(f'strip --strip-all {infile} -o {outfile} > /dev/null 2>&1')
        case 'arm32':
            ret_code = os.system(f'aarch64-linux-gnu-strip --strip-all {infile} -o {outfile} > /dev/null 2>&1')
        case 'arm64':
            ret_code = os.system(f'aarch64-linux-gnu-strip --strip-all {infile} -o {outfile} > /dev/null 2>&1')
        case 'mips32':
            ret_code = os.system(f'mips-linux-gnu-strip --strip-all {infile} -o {outfile} > /dev/null 2>&1')
        case 'mips64':
            ret_code = os.system(f'mips-linux-gnu-strip --strip-all {infile} -o {outfile} > /dev/null 2>&1')
        case _:
            raise ValueError(f"Cannot recognize the architecture of the provided file, whose e_machine is {arch}.")
    if ret_code != 0:
        raise RuntimeError(f"Meet error while trying to strip {infile}!")


def strip_file(infile, outfile):
    return strip_file_local(infile, outfile)
