from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from collections import defaultdict
import os


class BinaryBase(object):
    def __init__(self, original_file):
        self.original_file = original_file
        assert os.path.exists(original_file), f'{original_file} not exists'
        self.addr2name = self.extract_addr2name(self.original_file)

    def get_func_name(self, name, functions):
        if name not in functions:
            return name
        i = 0
        while True:
            new_name = name + '_' + str(i)
            if new_name not in functions:
                return new_name
            i += 1

    def scan_section(self, addr2name, section):
        """
        Function to extract function names from a shared library file.
        """
        if not section or not isinstance(section, SymbolTableSection) or section['sh_entsize'] == 0:
            return 0

        for nsym, symbol in enumerate(section.iter_symbols()):

            if symbol['st_info']['type'] == 'STT_FUNC' and symbol['st_shndx'] != 'SHN_UNDEF':
                func = symbol.name
                addr2name[symbol.entry['st_value']] = func

    def extract_addr2name(self, path):
        """
        return:
        """
        addr2name = {}
        with open(path, 'rb') as stream:
            elffile = ELFFile(stream)
            self.scan_section(addr2name, elffile.get_section_by_name('.symtab'))
            self.scan_section(addr2name, elffile.get_section_by_name('.dynsym'))
        return defaultdict(lambda: -1, addr2name)
