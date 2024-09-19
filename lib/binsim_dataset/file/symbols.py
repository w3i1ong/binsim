import os
import pickle
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

class FunctionSymbols:
    def __init__(self, file_path):
        self._file_path = file_path
        self._functions = {}
        self._source_files = []
        self._parse_file(file_path)

    def __getitem__(self, item: str):
        result: tuple|None = self._functions.get(item, None)
        if result is None:
            raise KeyError(f"Function {item} not found.")
        addr, size, file_idx, line_no = result
        return addr, size, self.source_files[file_idx], line_no

    def __contains__(self, item):
        return item in self._functions

    def __len__(self):
        return len(self._functions)

    def __iter__(self):
        return iter(self._functions.keys())

    def functions(self):
        return self._functions.keys()

    def items(self):
        return self._functions.items()

    def save(self, outfile):
        with open(outfile, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(infile):
        with open(infile, "rb") as f:
            return pickle.load(f)

    def _parse_file(self, file_path):
        with open(file_path, 'rb') as f:
            elffile = ELFFile(f)
            dwarfinfo = elffile.get_dwarf_info()

            addr_to_line, file_entries = {}, {}
            base = 0
            for CU in dwarfinfo.iter_CUs():
                lineprog = dwarfinfo.line_program_for_CU(CU)
                if lineprog:
                    for entry in lineprog.get_entries():
                        if entry.state:
                            addr_to_line[entry.state.address] = (entry.state.file + base - 1, entry.state.line)
                    for idx, file_entry in enumerate(lineprog['file_entry']):
                        dir_index = file_entry.dir_index
                        directory = lineprog['include_directory'][dir_index - 1].decode('utf-8', 'replace') if dir_index > 0 else ''
                        full_path = os.path.join(directory, file_entry.name.decode('utf-8', 'replace'))
                        file_entries[base + idx] = full_path
                    base += len(lineprog['file_entry'])
            srcfile2idx = {}
            for section in elffile.iter_sections():
                if isinstance(section, SymbolTableSection):
                    for symbol in section.iter_symbols():
                        if symbol['st_info']['type'] == 'STT_FUNC' and symbol['st_value'] != 0 and symbol['st_size'] != 0:
                            func_name = symbol.name
                            func_addr = symbol['st_value']
                            func_size = symbol['st_size']

                            file_name, line_no = "unknown", 0
                            if func_addr in addr_to_line:
                                file_index, line_no = addr_to_line[func_addr]
                                if file_index in file_entries:
                                    file_name = file_entries[file_index]
                            if file_name not in srcfile2idx:
                                srcfile2idx[file_name] = len(srcfile2idx)
                            self._functions[func_name] = (func_addr, func_size, srcfile2idx[file_name], line_no)
        self.source_files = list(srcfile2idx.keys())
