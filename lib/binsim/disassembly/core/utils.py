from enum import Enum


class Architecture(Enum):
    X86 = 'x86'
    X64 = 'x64'
    ARM32 = 'arm32'
    ARM64 = 'arm64'
    MIPS32 = 'mips32'
    MIPS64 = 'mips64'

    def __lt__(self, other):
        return self.value < other.value


ARCH2NORMALIZED_ARCH_NAME = {
    'x86': 'x86',
    'x86_64': 'x86_64',
    'armv7': 'arm32',
    'armv8': 'arm32',
    'aarch64': 'arm64',
    'mips32': 'mips32',
    'mips64': 'mips64',
    'mipsel32': 'mips32'
}


def get_normalized_arch_name(arch_name):
    match arch_name:
        case 'x86' | 'x64' | 'arm32' | 'arm64' | 'mips32' | 'mips64':
            return Architecture(arch_name)
        case 'x86_64':
            return Architecture.X64
        case 'armv7' | 'armv8':
            return Architecture.ARM32
        case 'aarch64':
            return Architecture.ARM64
        case 'mipsel32':
            return Architecture.MIPS32
        case _:
            raise KeyError(f"Unknown architecture: {arch_name}")
