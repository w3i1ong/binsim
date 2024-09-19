import os
import hashlib
import queue
import pickle
import logging
import threading
import binaryninja
from tqdm import tqdm
import multiprocessing
from typing import Union, List
from abc import ABC, abstractmethod
from binaryninja import BinaryView, Function
from binsim.disassembly.utils import strip_file, get_architecture
from binsim.disassembly.utils import BinaryBase
from binsim.utils import init_logger

logger = init_logger("DisassemblerBase", level=logging.INFO,
                     console=True, console_level=logging.INFO)

class DisassemblerBase(ABC):
    def __init__(self, normalizer):
        self.normalizer = normalizer

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
                        need_strip=True):
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
        :return:
        """
        assert db_file is None or db_file.endswith('.bndb'), "The database file must be a bndb file."
        filename = os.path.abspath(filename)

        binary = None
        if (db_file is not None and os.path.exists(db_file)) and not regenerate:
            # sometimes, the db_file may be corrupted which may cause an exception.
            # we need to catch the exception and delete the corrupted file.
            try:
                binary = binaryninja.load(db_file)
            except Exception as e:
                os.remove(db_file)

        if binary is None:
            if need_strip:
                stripped_file = f'/tmp/{hashlib.md5(filename.encode()).hexdigest()}.stripped'

                try:
                    strip_file(filename, stripped_file)
                except RuntimeError as e:
                    logger.error(f"Meet error while trying to strip {filename}")
                    return

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

    @abstractmethod
    def _visit_functions(self, binary: BinaryView,
                         load_pdb,
                         keep_thunks=False,
                         keep_unnamed=False,
                         verbose=True,
                         keep_large=False,
                         large_ins_threshold=1000,
                         large_graph_threshold=1000,
                         keep_small=False,
                         small_ins_threshold=10,
                         small_graph_threshold=10):
        pass

    def disassemble(self,
                    filename,
                    outfile,
                    db_file=None,
                    load_pdb=False,
                    regenerate=False,
                    reanalysis=False,
                    keep_thunks=False,
                    keep_unnamed=False,
                    verbose=True,
                    keep_large=False,
                    large_ins_threshold=10000,
                    large_graph_threshold=1000,
                    keep_small=False,
                    small_ins_threshold=10,
                    small_graph_threshold=3):
        if os.path.exists(outfile) and not reanalysis:
            return

        try:
            get_architecture(filename)
        except ValueError:
            logger.error(f"Cannot get arch of {filename}, skipped")
            return
        # check output file
        if os.path.isdir(outfile):
            raise ValueError("The output file cannot be a directory.")
        out_dir = os.path.split(outfile)[0]
        if out_dir != '' and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # check database file
        if db_file is not None:
            if os.path.isdir(db_file):
                raise ValueError("The database file cannot be a directory.")
            db_dir = os.path.split(db_file)[0]
            if db_dir != '' and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
        if not os.path.exists(outfile) or reanalysis:
            cfgs = self.visit_functions(filename, db_file=db_file, load_pdb=load_pdb, regenerate=regenerate,
                                        keep_thunks=keep_thunks, keep_unnamed=keep_unnamed, verbose=verbose,
                                        keep_large=keep_large, large_ins_threshold=large_ins_threshold,
                                        large_graph_threshold=large_graph_threshold, keep_small=keep_small,
                                        small_ins_threshold=small_ins_threshold,
                                        small_graph_threshold=small_graph_threshold)
            if cfgs:
                # disassemble
                with open(outfile, 'wb') as f:
                    pickle.dump(cfgs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def disassemble_wrapper(self, kwargs):
        self.disassemble(**kwargs)

    def disassemble_files(self,
                          src_files: Union[str, List[str]],
                          out_files: Union[str, List[str]],
                          db_files: Union[str, List[str]] = None,
                          workers=0,
                          verbose=True,
                          load_pdb=False,
                          regenerate=False,
                          reanalysis=False,
                          keep_thunks=False,
                          keep_unnamed=False,
                          keep_large=False,
                          large_ins_threshold=1000,
                          large_graph_threshold=1000,
                          keep_small=False,
                          small_ins_threshold=10,
                          small_graph_threshold=10):
        assert isinstance(src_files, list), "The input files must be a list of files."
        assert isinstance(out_files, list), "The output files must be a list of files."
        assert isinstance(db_files, list) or db_files is None, "The database files must be a list of files."
        if db_files is None:
            db_files = [None] * len(src_files)
        assert len(src_files) == len(
            out_files), "The number of input files must be equal to the number of output files."
        assert len(src_files) == len(
            db_files), "The number of input files must be equal to the number of database files."
        assert workers >= 0, "The number of workers must be greater than or equal to 0."

        def _gen_args():
            for i in range(len(src_files)):
                yield {
                    'filename': src_files[i],
                    'outfile': out_files[i],
                    'db_file': db_files[i],
                    'load_pdb': load_pdb,
                    'regenerate': regenerate,
                    'reanalysis': reanalysis,
                    'keep_thunks': keep_thunks,
                    'verbose': verbose,
                    'keep_unnamed': keep_unnamed,
                    'keep_large': keep_large,
                    'large_ins_threshold': large_ins_threshold,
                    'large_graph_threshold': large_graph_threshold,
                    'keep_small': keep_small,
                    'small_ins_threshold': small_ins_threshold,
                    'small_graph_threshold': small_graph_threshold
                }

        kwargs_itr = _gen_args()
        if workers == 0:
            for kwargs in tqdm(kwargs_itr, total=len(src_files)):
                self.disassemble(**kwargs)
        else:
            with multiprocessing.Pool(processes=workers, maxtasksperchild=5) as pool:
                for _ in tqdm(pool.imap_unordered(self.disassemble_wrapper, kwargs_itr), total=len(src_files)):
                    pass
