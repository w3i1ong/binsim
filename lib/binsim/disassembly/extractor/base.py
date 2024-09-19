import os
import shutil

import filelock
import rocksdb
import pickle
import hashlib
import multiprocessing
from abc import ABC
from tqdm import tqdm
from filelock import FileLock
from typing import Union, List

class ExtractorBase(ABC):
    def __init__(self, disassembler,
                 workers=0,
                 verbose=True,
                 load_pdb=False,
                 regenerate=False,
                 keep_thunk=False,
                 keep_unnamed=False,
                 keep_large=False,
                 large_ins_threshold=3000,
                 large_graph_threshold=300,
                 keep_small=False,
                 small_ins_threshold=10,
                 small_graph_threshold=5,
                 debug=False,
                 checkpoint=True,
                 incremental=False,
                 neural_input_kwargs=None,
                 disassemble_kwargs=None):
        self._disassembler = disassembler
        self._workers = workers
        self._verbose = verbose
        self._load_pdb = load_pdb
        self._regenerate = regenerate
        self._keep_thunks = keep_thunk
        self._keep_unnamed = keep_unnamed
        self._keep_large = keep_large
        self._large_ins_threshold = large_ins_threshold
        self._large_graph_threshold = large_graph_threshold
        self._keep_small = keep_small
        self._small_ins_threshold = small_ins_threshold
        self._small_graph_threshold = small_graph_threshold
        self._debug = debug
        self._disassemble_kwargs = disassemble_kwargs if disassemble_kwargs is not None else {}
        self.__as_neural_input_kwargs = neural_input_kwargs if neural_input_kwargs is not None else {}
        self._incremental = incremental
        self._checkpoint = checkpoint

    def before_extract_process(self):
        pass

    def after_extract_process(self):
        pass

    def disassemble_files(self,
                          src_files: Union[str, List[str]],
                          out_files: Union[str, List[str]],
                          db_files: Union[str, List[str]] = None,
                          dataset_dir:str = None):
        assert isinstance(src_files, list), "The input files must be a list of files."
        assert isinstance(out_files, list), "The output files must be a list of files."
        assert isinstance(db_files, list) or db_files is None, "The database files must be a list of files."
        if db_files is None:
            db_files = [None] * len(src_files)
        assert len(src_files) == len(out_files) and len(src_files) == len(db_files), \
            "The number of source files, output files, and database files must be equal, but got " \
            f"{len(src_files)}, {len(out_files)}, and {len(db_files)} respectively."
        assert self._workers >= 0, "The number of workers must be greater than or equal to 0."

        remove_old_dataset = os.path.exists(dataset_dir) and not self._incremental
        checkpoint_record = f'{dataset_dir}/binsim-checkpoint'
        if not self._checkpoint or not os.path.exists(checkpoint_record):
            for out_file in out_files:
                if os.path.exists(out_file):
                    os.unlink(out_file)
        else:
            with open(checkpoint_record, 'r') as f:
                checkpoint_files = set(f.read().splitlines())
            filtered_src_files, filtered_out_files, filtered_db_files = [], [], []
            for src_file, out_file, db_file in zip(src_files, out_files, db_files):
                if src_file not in checkpoint_files:
                    filtered_src_files.append(src_file)
                    filtered_out_files.append(out_file)
                    filtered_db_files.append(db_file)
            if len(src_files) == len(filtered_src_files):
                remove_old_dataset &= True
            src_files, out_files, db_files = filtered_src_files, filtered_out_files, filtered_db_files

        if remove_old_dataset:
            shutil.rmtree(dataset_dir)
        def _gen_args():
            for i in range(len(src_files)):
                yield {
                    'filename': src_files[i],
                    'outfile': out_files[i],
                    'db_file': db_files[i],
                    'load_pdb': self._load_pdb,
                    'regenerate': self._regenerate,
                    'keep_thunks': self._keep_thunks,
                    'verbose': self._verbose,
                    'keep_unnamed': self._keep_unnamed,
                    'keep_large': self._keep_large,
                    'large_ins_threshold': self._large_ins_threshold,
                    'large_graph_threshold': self._large_graph_threshold,
                    'keep_small': self._keep_small,
                    'small_ins_threshold': self._small_ins_threshold,
                    'small_graph_threshold': self._small_graph_threshold,
                    "dataset_dir": dataset_dir,
                    "debug": self._debug,
                    "with_ui": self._debug,
                    **self._disassemble_kwargs
                }

        self.before_extract_process()
        kwargs_itr = _gen_args()
        if self._workers == 0:
            for kwargs in tqdm(kwargs_itr, total=len(src_files)):
                self.disassemble_wrapper(kwargs)
        else:
            with multiprocessing.Pool(processes=self._workers, maxtasksperchild=1) as pool:
                for _ in tqdm(pool.imap_unordered(self.disassemble_wrapper, kwargs_itr), total=len(src_files)):
                    pass
        self.after_extract_process()
        if os.path.exists(f'{dataset_dir}.lock'):
            os.unlink(f'{dataset_dir}.lock')

    def disassemble_wrapper(self, kwargs):
        return self.disassemble(**kwargs)

    def after_disassemble(self, cfgs):
        return cfgs

    def disassemble(self, dataset_dir, outfile, debug, **kwargs):
        cfgs = self._disassembler.disassemble(**kwargs)
        cfgs = self.after_disassemble(cfgs)
        out_dir = os.path.split(outfile)[0]
        os.makedirs(out_dir, exist_ok=True)
        if cfgs:
            if dataset_dir is not None:
                results, new_cfgs = [], []
                for cfg in cfgs:
                    content = cfg.as_neural_input_raw(**self.__as_neural_input_kwargs)
                    if content is None:
                        continue
                    content_hash = hashlib.md5(content).hexdigest()
                    results.append((content_hash, content))
                    cfg.content_hash = content_hash
                    new_cfgs.append(cfg)
                cfgs = new_cfgs
                lock_file = f'{dataset_dir}.lock'
                with FileLock(lock_file):
                    db = rocksdb.DB(dataset_dir, rocksdb.Options(create_if_missing=True))
                    for content_hash, data in results:
                        db.put(content_hash.encode(), data)
                    del db

            if not debug:
                cfgs = [cfg.minimize() for cfg in cfgs]
            with open(outfile, 'wb') as f:
                pickle.dump(cfgs, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self._checkpoint:
                checkpoint_record = f'{dataset_dir}/binsim-checkpoint'
                with filelock.FileLock(f"{dataset_dir}.lock"):
                    with open(checkpoint_record, 'a') as f:
                        f.write(f'{kwargs["filename"]}\n')
