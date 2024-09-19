import os
import logging
from abc import ABC, abstractmethod
from binsim.disassembly.utils import get_architecture
from binsim.utils import init_logger

logger = init_logger("DisassemblerBase", level=logging.INFO,
                     console=True, console_level=logging.INFO)

class DisassemblerBase(ABC):
    def __init__(self, normalizer):
        self.normalizer = normalizer

    @abstractmethod
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
                        small_graph_threshold=5,
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
        pass

    @abstractmethod
    def _visit_functions(self, binary,
                         load_pdb,
                         keep_thunks=False,
                         keep_unnamed=False,
                         verbose=True,
                         keep_large=False,
                         large_ins_threshold=1000,
                         large_graph_threshold=1000,
                         keep_small=False,
                         small_ins_threshold=10,
                         small_graph_threshold=10, **kwargs):
        pass

    def disassemble(self,
                    filename,
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
                    **kwargs):
        try:
            get_architecture(filename)
        except ValueError:
            logger.error(f"Cannot get arch of {filename}, skipped")
            return
        # check database file
        if db_file is not None:
            if os.path.isdir(db_file):
                raise ValueError("The database file cannot be a directory.")
            db_dir = os.path.split(db_file)[0]
            if db_dir != '' and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
        cfgs = self.visit_functions(filename, db_file=db_file, load_pdb=load_pdb, regenerate=regenerate,
                                    keep_thunks=keep_thunks, keep_unnamed=keep_unnamed, verbose=verbose,
                                    keep_large=keep_large, large_ins_threshold=large_ins_threshold,
                                    large_graph_threshold=large_graph_threshold, keep_small=keep_small,
                                    small_ins_threshold=small_ins_threshold,
                                    small_graph_threshold=small_graph_threshold)
        return cfgs
