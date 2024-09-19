from .JTransSeq import JTransSeq
from binsim.disassembly.backend.ida.base import IDANormalizerBase
from binsim.disassembly.backend.ida.graph.jTransSeq.util import load_ins_parser

class JTransNormalizer(IDANormalizerBase):
    def __init__(self, arch, max_length):
        super().__init__(arch)
        self._max_length = max_length
        self.asm_parser = load_ins_parser()

    def __call__(self, func)->JTransSeq:
        import idc
        cfg, func_hash = self.get_cfg(func)
        nodes = list(cfg.nodes())
        nodes.sort()
        asm, bb_len = [], []
        for node in nodes:
            asm.extend(cfg.nodes[node]['asm'])
            bb_len.append((node, len(cfg.nodes[node]['asm'])))
        func_name, token_id_list, func_hash, node_num = (
            idc.get_name(func.start_ea), self.asm_parser.parseFunction(asm, max_length=self._max_length, bb_length=bb_len),
            func_hash, cfg.number_of_nodes())
        return JTransSeq(func_name, self.arch, token_id_list, node_num=node_num, func_hash=func_hash, ins_num=len(asm))

    def get_cfg(self, func):
        import idc, idaapi, idautils
        import networkx as nx
        from binsim.disassembly.backend.ida.graph.utils import compute_function_hash
        def get_attr(block, func_addr_set):
            asm = []
            curr_addr = block.start_ea
            if curr_addr not in func_addr_set:
                return None
            while curr_addr <= block.end_ea:
                asm.append(idc.GetDisasm(curr_addr))
                curr_addr = idc.next_head(curr_addr, block.end_ea)
            return asm

        nx_graph = nx.DiGraph()
        flowchart = idaapi.FlowChart(func, flags=idaapi.FC_PREDS)
        func_hash = compute_function_hash(flowchart)
        func_addr_set = set([addr for addr in idautils.FuncItems(func.start_ea)])
        for block in flowchart:
            # Make sure all nodes are added (including edge-less nodes)
            asm = get_attr(block, func_addr_set)
            if asm is None:
                continue
            nx_graph.add_node(block.start_ea, asm=asm)
            for pred in block.preds():
                if pred.start_ea not in func_addr_set:
                    continue
                nx_graph.add_edge(pred.start_ea, block.start_ea)
            for succ in block.succs():
                if succ.start_ea not in func_addr_set:
                    continue
                nx_graph.add_edge(block.start_ea, succ.start_ea)
        return nx_graph, func_hash
    

