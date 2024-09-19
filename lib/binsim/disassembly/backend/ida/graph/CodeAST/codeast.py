import dgl
import torch
from binsim.disassembly.core import BinsimFunction
from typing import List, Tuple, Any


class CodeAST(BinsimFunction):
    def __init__(self, name,
                 arch,
                 features: List[Any],
                 ast_edges: List[Tuple[int, int]],
                 callee_num: int,
                 node_num: int,
                 ins_num: int,
                 func_hash=None):
        super().__init__(name, func_hash=func_hash, func_arch=arch, node_num=node_num, ins_num=ins_num)
        self._features = features
        self._ast_edges = ast_edges
        self._callee_num = callee_num

    def __str__(self):
        return f'<CodeAST::{self.name}' \
               f'(node_num={len(self._features)})>'

    def as_neural_input(self) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
        U, V = zip(*self._ast_edges)
        node_features = [f[0] for f in self._features]
        return dgl.graph((U, V)), torch.tensor(node_features, dtype=torch.long), self._callee_num

    def as_neural_input_raw(self, *args, **kwargs):
        from binsim.disassembly.backend.binaryninja.graph.datautils import code_ast_pack_neural_input
        src, dst = zip(*self._ast_edges)
        features = [f[0] for f in self._features]
        return code_ast_pack_neural_input(src, dst, len(features), self._callee_num, features)

    @staticmethod
    def collate_raw_neural_input(inputs: List[bytes], **kwargs):
        from binsim.disassembly.backend.binaryninja.graph.datautils import code_ast_collate_neural_input
        return code_ast_collate_neural_input(inputs)

    def features(self):
        return self._features, self._ast_edges

    def minimize(self):
        self._features = self._ast_edges = self._callee_num = None
        return self


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.op = None  #
        self.value = None  #
        self.opname = ""  #

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        try:
            if getattr(self, '_size'):
                return self._size
        except AttributeError as e:
            count = 1
            for i in range(self.num_children):
                count += self.children[i].size()
            self._size = count
            return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __str__(self):
        return self.opname

    def _dfs(self, node_id, node_features: list, edges: list) -> int:
        node_features.append((self.op, self.opname, self.value))
        old_node_id = node_id
        for child in self.children:
            edges.append((node_id + 1, old_node_id))
            node_id = child._dfs(node_id + 1, node_features, edges)
        return node_id

    def as_code_ast(self, name, arch, func_hash, callee_num, node_num, ins_num) -> CodeAST:
        node_features, edges = [], []
        self._dfs(0, node_features, edges)
        return CodeAST(name,
                       arch=arch,
                       callee_num=callee_num,
                       features=node_features,
                       ast_edges=edges,
                       func_hash=func_hash,
                       node_num=node_num,
                       ins_num=ins_num)
