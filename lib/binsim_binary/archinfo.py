from enum import Enum

normalized_arch = {
    "arm": "arm32",
    "arm32": "arm32",
    "arm64": "arm64",
    "aarch64": "arm64",
    "mips64": "mips64",
    "mips32": "mips32",
    "mips": "mips32",
    "x86": "x86",
    "i386": "x86",
    "i86": "x86",
    "i686": "x86",
    "x64": "x64",
    "amd64": "x64",
    "x86_64": "x64",
}

class Arch(Enum):
    arm32 = 'arm32'
    arm64 = 'arm64'
    mips32 = 'mips32'
    mips64 = 'mips64'
    x86 = 'x86'
    x64 = 'x64'

    def __str__(self):
        return f"<Arch.{self.value}>"

    def __repr__(self):
        return f"<Arch.{self.value}>"

    @staticmethod
    def from_string(arch_str: str) -> 'Arch':
        if arch_str in ['arm32', 'arm64', 'mips32', 'mips64', 'x86', 'x64']:
            return Arch(arch_str)
        if '-' in arch_str:
            arch, platform, _ = arch_str.split('-')
            arch = normalized_arch.get(arch, None)
            if arch is not None:
                return Arch(arch)
        raise ValueError(f"Cannot recognize the architecture: {arch_str}")

    @property
    def prefix(self):
        match self.value:
            case 'arm32':
                return 'arm-linux-gnueabi'
            case 'arm64':
                return 'aarch64-linux-gnu'
            case 'mips32':
                return 'mips-linux-gnu'
            case 'mips64':
                return 'mips64-linux-gnuabi64'
            case 'x86':
                return 'i686-linux-gnu'
            case 'x64':
                return 'x86_64-linux-gnu'
