from binsim.disassembly.binaryninja.base import DisassemblerBase
from binsim.disassembly.binaryninja.base import NormalizerBase
from binsim.disassembly.binaryninja.base import CFGBase
from binaryninja import BinaryView
from typing import List, Type


class BinaryNinja(DisassemblerBase):
    def __init__(self, normalizer: Type[NormalizerBase], normalizer_kwargs=None, **kwargs):
        super(BinaryNinja, self).__init__(normalizer)
        if normalizer_kwargs is None:
            normalizer_kwargs = dict()
        self._normalizers = dict()
        self._normalizer_kwargs = normalizer_kwargs

    def get_normalizer(self, arch: str) -> NormalizerBase:
        if arch not in self._normalizers:
            self._normalizers[arch] = self.normalizer(arch, **self._normalizer_kwargs)
        return self._normalizers[arch]

    def _visit_functions(self, binary: BinaryView,
                         load_pdb=False,
                         keep_thunks=False,
                         keep_unnamed=False,
                         verbose=True,
                         keep_large=False,
                         large_ins_threshold=1000,
                         large_graph_threshold=300,
                         keep_small=False,
                         small_ins_threshold=10,
                         small_graph_threshold=3) -> List[CFGBase]:
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
