# encoding=utf-8
"""
python3
A script for facilitating the usage of ast_generator.py
"""
import sys, os, logging
from tqdm import tqdm
from typing import Union, List
from multiprocessing import Pool
from .wrapper import execute_ida_script
from binsim.disassembly.ida.ast import script as ast_script
from binsim.disassembly.ida.cfg import script as cfg_script
from binsim.disassembly.utils import strip_file
from binsim.disassembly.utils.globals import GraphType

platform = sys.platform

logger = logging.getLogger("IDADisassembler")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


class IDADisassembler:
    def __init__(self, normalizer: str, ida_path: str):
        self._normalizer = normalizer
        self._ida_path = ida_path
        assert os.path.isdir(ida_path), "You must provide the root directory of IDA."
        assert os.path.exists(os.path.join(ida_path, "ida64.exe")) and os.path.exists(
            os.path.join(ida_path, "ida.exe")), "Cannot find ida.exe or ida64.exe in the provided directory."

    def disassemble(self,
                    filename,
                    outfile,
                    db_file,
                    log_file=None,
                    load_pdb=False,
                    regenerate=False,
                    reanalysis=False,
                    keep_thunks=False,
                    keep_unnamed=False,
                    verbose=True,
                    timeout=10,
                    debug=False,
                    keep_large=False,
                    large_ins_threshold=100000,
                    large_graph_threshold=100000,
                    keep_small=False,
                    small_ins_threshold=10,
                    small_graph_threshold=3,
                    max_length=512):
        # check output file
        if os.path.isdir(outfile):
            raise ValueError("The output file cannot be a directory.")
        # check database file
        if os.path.isdir(db_file):
            raise ValueError("The database file cannot be a directory.")
        # check log file
        if log_file is not None and os.path.isdir(log_file):
            raise ValueError("The log file cannot be a directory.")

        out_dir = os.path.split(outfile)[0]
        if out_dir != '' and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        db_dir = os.path.split(db_file)[0]
        if db_dir != '' and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        if log_file is not None:
            log_dir = os.path.split(log_file)[0]
            if log_dir != '' and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

        # 1. generate striped file
        stripped_file = os.path.splitext(db_file)[0] + '.strip'
        # 2. only x86/x64 is supported
        try:
            arch = self.get_arch(filename)
        except ValueError as e:
            logger.error(f"Cannot get arch of {filename}, skipped")
            return
        if not reanalysis and os.path.exists(outfile):
            return
        strip_file(filename, stripped_file)
        # 2. extract function address
        filename = os.path.abspath(filename)
        # 3. get proper disassembler
        ida = self.get_proper_ida_exe(filename)
        match self._normalizer:
            case GraphType.CodeAST:
                execute_ida_script(ida,
                                   script_path=ast_script.entry_point,
                                   binary=stripped_file,
                                   out_file=outfile,
                                   db_file=db_file,
                                   logfile=log_file,
                                   function="",
                                   timeout=timeout,
                                   original_file=filename,
                                   regenerate=regenerate,
                                   reanalysis=reanalysis,
                                   with_ui=debug,
                                   keep_thunk=keep_thunks,
                                   keep_unnamed=keep_unnamed,
                                   keep_small=keep_small,
                                   small_ins_threshold=small_ins_threshold,
                                   small_graph_threshold=small_graph_threshold,
                                   keep_large=keep_large,
                                   large_ins_threshold=large_ins_threshold,
                                   large_graph_threshold=large_graph_threshold)
            case GraphType.JTransSeq:
                if arch != 'x64':
                    logger.error(f"JTrans only supports x64. But {filename} is of {arch}, skipped")
                    return
                execute_ida_script(ida,
                                   script_path=cfg_script.entry_point,
                                   binary=stripped_file,
                                   out_file=outfile,
                                   db_file=db_file,
                                   logfile=log_file,
                                   timeout=timeout,
                                   with_ui=debug,
                                   keep_thunk=keep_thunks,
                                   keep_unnamed=keep_unnamed,
                                   keep_small=keep_small,
                                   small_ins_threshold=small_ins_threshold,
                                   small_graph_threshold=small_graph_threshold,
                                   keep_large=keep_large,
                                   large_ins_threshold=large_ins_threshold,
                                   large_graph_threshold=large_graph_threshold,
                                   original_file=filename,
                                   max_length=max_length,
                                   regenerate=regenerate,
                                   reanalysis=reanalysis,
                                   arch=arch)
            case ukn_type:
                raise ValueError(f"Meet unsupported normalizer type {ukn_type}, while disassemble file with IDA.")

    def get_proper_ida_exe(self, filename):
        with open(filename, 'rb') as f:
            data = f.read(16)
        if data.startswith(b'\x7fELF'):
            if data[4] == 1:
                return os.path.join(self._ida_path, "ida.exe")
            elif data[4] == 2:
                return os.path.join(self._ida_path, "ida64.exe")
            raise ValueError("The provided file is neither a 32-bit ELF file nor a 64-bit ELF file.")
        elif filename.endswith("idb"):
            return os.path.join(self._ida_path, "ida.exe")
        elif filename.endswith("i64"):
            return os.path.join(self._ida_path, "ida64.exe")
        raise ValueError("The provided file is neither an ELF file nor an IDA database file.")

    def get_arch(self, filename):
        with open(filename, 'rb') as f:
            data = f.read(16)
            if not data.startswith(b'\x7fELF'):
                raise ValueError("The provided file is not an ELF file.")
            if data[4] == 1:
                data = f.read(4)[2:]
                if data == b'\x03\x00':
                    return 'x86'
                elif data == b'\x28\x00':
                    return 'arm'
                return 'ukn'
            elif data[4] == 2:
                data = f.read(4)[2:]
                if data == b'\x3e\x00':
                    return 'x64'
                elif data == b'\x28\x00':
                    return 'arm64'
                return 'ukn'

    def _disassemble_file_worker(self, kwargs):
        return self.disassemble(**kwargs)

    def disassemble_files(self,
                          src_files: Union[str, List[str]],
                          out_files: Union[str, List[str]],
                          db_files: Union[str, List[str]],
                          log_files: Union[str, List[str]] = None,
                          workers=0,
                          verbose=True,
                          load_pdb=False,
                          regenerate=False,
                          keep_thunks=False,
                          keep_unnamed=False,
                          keep_large=False,
                          large_ins_threshold=1000,
                          large_graph_threshold=1000,
                          keep_small=False,
                          small_ins_threshold=10,
                          small_graph_threshold=10,
                          **extra_kwargs):
        assert isinstance(src_files, list), "The input files must be a list of files."
        assert isinstance(out_files, list), "The output files must be a list of files."
        assert isinstance(db_files, list) or db_files is None, "The database files must be a list of files."
        assert isinstance(log_files, list) or log_files is None, "The log files must be a list of files."
        if db_files is None:
            db_files = [None] * len(src_files)
        if log_files is None:
            log_files = [None] * len(src_files)
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
                    'log_file': log_files[i],
                    'load_pdb': load_pdb,
                    'regenerate': regenerate,
                    'keep_thunks': keep_thunks,
                    'verbose': verbose,
                    'keep_unnamed': keep_unnamed,
                    'keep_large': keep_large,
                    'large_ins_threshold': large_ins_threshold,
                    'large_graph_threshold': large_graph_threshold,
                    'keep_small': keep_small,
                    'small_ins_threshold': small_ins_threshold,
                    'small_graph_threshold': small_graph_threshold,
                    **extra_kwargs
                }

        kwargs_itr = _gen_args()
        if workers == 0:
            for idx, kwargs in tqdm(enumerate(kwargs_itr), total=len(src_files)):
                self.disassemble(**kwargs)
        else:
            pool = Pool(processes=workers)
            for _ in tqdm(pool.imap_unordered(self._disassemble_file_worker, kwargs_itr), total=len(src_files)):
                pass
            pool.close()
