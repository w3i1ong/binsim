import os
import hashlib
from binsim.utils import init_logger
from typing import List, Type, TYPE_CHECKING
from binsim.disassembly.utils import BinaryBase
from binsim.disassembly.utils import strip_file
from binsim.disassembly.core import DisassemblerBase
from binsim.disassembly.core import NormalizerBase
from binsim.disassembly.core import BinsimFunction

if TYPE_CHECKING:
    from binaryninja import BinaryView


logger = init_logger("BinaryNinja", level='DEBUG', console=True, console_level='DEBUG')

class BinaryNinja(DisassemblerBase):
    def __init__(self, normalizer: Type[NormalizerBase], normalizer_kwargs=None):
        super(BinaryNinja, self).__init__(normalizer)
        if normalizer_kwargs is None:
            normalizer_kwargs = dict()
        self._normalizers = dict()
        self._normalizer_kwargs = normalizer_kwargs

    def visit_functions(self, filename,
                        db_file=None,
                        load_pdb=False,
                        regenerate=False,
                        keep_thunks=False,
                        keep_unnamed=False,
                        verbose=True,
                        keep_large=False,
                        large_ins_threshold=10000,
                        large_graph_threshold=1000,
                        keep_small=False,
                        small_ins_threshold=10,
                        small_graph_threshold=3,
                        need_strip=True,
                        **kwargs):
        """
        Disassemble a binary file with Binary Ninja.
        :param filename: The file to be disassembled.
        :param db_file: A bndb file to store the disassembled result. If the file exists, the disassembler will load it
            instead of disassembling the binary file.
        :param load_pdb: Whether to load PDB file.
        :param regenerate: Whether to regenerate the database file, if the database file exists.
        :param keep_thunks: whether to keep thunk functions
        :param keep_unnamed:
        :param verbose: Whether to show the progress bar.
        :param keep_large: Whether to keep large functions.
        :param large_ins_threshold: The threshold of the number of instructions in a function. If keep_large is False,
            functions with more than large_ins_threshold instructions will be discarded.
        :param large_graph_threshold: The threshold of the number of basic blocks in a function. If keep_large is False,
        :param keep_small: Whether to keep small functions.
        :param small_ins_threshold: The threshold of the number of instructions in a function. If keep_small is False,
        :param small_graph_threshold:  The threshold of the number of basic blocks in a function. If keep_small is False,
        :param need_strip: Whether to strip the binary file before disassembling.
        :return:
        """
        import binaryninja
        assert db_file is None or db_file.endswith('.bndb'), "The database file must be a bndb file."
        filename = os.path.abspath(filename)

        binary = None
        if (db_file is not None and os.path.exists(db_file)) and not regenerate:
            # sometimes, the db_file may be corrupted which may cause an exception.
            # we need to catch the exception and delete the corrupted file.
            try:
                binary = binaryninja.load(db_file)
            except Exception:
                os.remove(db_file)

        if binary is None:
            if need_strip:
                stripped_file = f'/tmp/{hashlib.md5(filename.encode()).hexdigest()}.stripped'

                try:
                    strip_file(filename, stripped_file)
                except ValueError as e:
                    logger.error(f"Meet error while trying to strip {filename}. Error message: {e}.")
                    return
                except RuntimeError as e:
                    logger.warning(f"Cannot strip {filename}. Just use the original file.")
                    stripped_file = filename

                temp = BinaryBase(filename)
                try:
                    binary = binaryninja.load(stripped_file)
                except:
                    logger.error(f"Unable to create new BinaryView for {stripped_file}")
                for k, v in temp.addr2name.items():
                    binary.add_function(k)
                    func = binary.get_function_at(k)
                    if func is not None:
                        func.name = v
            else:
                binary = binaryninja.load(filename)
            if db_file is not None:
                binary.create_database(db_file)
        result = self._visit_functions(binary,
                                       load_pdb,
                                       keep_thunks=keep_thunks,
                                       keep_unnamed=keep_unnamed,
                                       verbose=verbose,
                                       keep_large=keep_large,
                                       large_ins_threshold=large_ins_threshold,
                                       large_graph_threshold=large_graph_threshold,
                                       keep_small=keep_small,
                                       small_ins_threshold=small_ins_threshold,
                                       small_graph_threshold=small_graph_threshold)
        return result

    def get_normalizer(self, arch: str) -> NormalizerBase:
        if arch not in self._normalizers:
            self._normalizers[arch] = self.normalizer(arch, **self._normalizer_kwargs)
        return self._normalizers[arch]

    def _visit_functions(self, binary: "BinaryView",
                         load_pdb=False,
                         keep_thunks=False,
                         keep_unnamed=False,
                         verbose=True,
                         keep_large=False,
                         large_ins_threshold=1000,
                         large_graph_threshold=300,
                         keep_small=False,
                         small_ins_threshold=10,
                         small_graph_threshold=3, **kwargs) -> List[BinsimFunction]:
        """
        Disassemble a binary file with Binary Ninja.
        :param binary: The binary view to be iterated.
        :param load_pdb: Whether to load PDB file.
        :param keep_thunks: whether to keep thunk functions
        :param keep_unnamed: whether to keep unnamed functions
        :param verbose: Whether to show the progress bar.
        :param keep_large: Whether to keep large functions.
        :param large_ins_threshold: The threshold of the number of instructions in a function. If keep_large is False,
            functions with more than large_ins_threshold instructions will be discarded.
        :param large_graph_threshold: The threshold of the number of basic blocks in a function. If keep_large is False,
            functions with more than large_graph_threshold basic blocks will be discarded.
        :param keep_small: Whether to keep small functions.
        :param small_ins_threshold: The threshold of the number of instructions in a function. If keep_small is False,
            functions with less than small_ins_threshold instructions will be discarded.
        :param small_graph_threshold: The threshold of the number of basic blocks in a function. If keep_small is False,
            functions with less than small_graph_threshold basic blocks will be discarded.
        :return:
        """
        import binaryninja
        binaryninja.disable_default_log()
        binaryninja.log.close_logs()
        normalizer = self.get_normalizer(binary.arch.name)
        cfg_list = []
        functions = binary.functions
        for function in functions:
            if not keep_thunks and function.is_thunk:
                continue
            # todo: it seems that binaryninja provides a better way to check if a function is unnamed
            if not keep_unnamed and function.name.startswith('sub_'):
                continue

            instruction_num = len(list(function.instructions))
            bb_num = len(list(function.basic_blocks))

            if not keep_large and (instruction_num > large_ins_threshold or bb_num > large_graph_threshold):
                continue

            if not keep_small and (instruction_num < small_ins_threshold or bb_num < small_graph_threshold):
                continue
            cfg_list.append(normalizer(function))
        return cfg_list
