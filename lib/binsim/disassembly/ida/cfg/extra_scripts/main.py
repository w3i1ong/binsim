import idc
import idautils
import idaapi
import logging
import pickle
import networkx as nx
from hashlib import sha256
from binsim.disassembly.utils import BinaryBase
from binsim.disassembly.ida.cfg.util import IDAIns
from binsim.disassembly.ida.cfg import JTransSeq
from binsim.disassembly.ida.utils import HeartbeatClient


def compute_function_hash(flowchart):
    operators = []
    for bb in sorted(list(flowchart), key=lambda x: x.start_ea):
        ea = bb.start_ea
        while bb.start_ea <= ea < bb.end_ea:
            # get the mnemonic of the instruction
            mnem = print_insn_mnem(ea)
            operators.append(mnem)
            for op_idx in range(8):
                operand_type = get_operand_type(ea, op_idx)
                if operand_type == 0:
                    break
                elif operand_type == 2:
                    operators.append('[ADDR]')
                elif operand_type == 3 or operand_type == 4:
                    operators.append('[MEM]')
                elif operand_type == 5:
                    operators.append('[IMM]')
                elif operand_type == 6:
                    operators.append('[FAR]')
                elif operand_type == 7:
                    operators.append('[NEAR]')
                else:
                    operators.append(print_operand(ea, op_idx))
            ea = idc.next_head(ea)
    return sha256("".join(operators).encode()).hexdigest()


class BinaryData(BinaryBase):
    def __init__(self, unstrip_path):
        super(BinaryData, self).__init__(unstrip_path)
        self.fix_up()

    def fix_up(self):
        for addr in self.addr2name:
            # incase some functions' instructions are not recognized by IDA
            idc.create_insn(addr)
            idc.add_func(addr)

    def get_asm(self, func):
        instGenerator = idautils.FuncItems(func)
        asm_list = []
        for inst in instGenerator:
            asm_list.append(idc.GetDisasm(inst))
        return asm_list

    def get_rawbytes(self, func):
        instGenerator = idautils.FuncItems(func)
        rawbytes_list = b""
        for inst in instGenerator:
            rawbytes_list += idc.get_bytes(inst, idc.get_item_size(inst))
        return rawbytes_list

    def get_cfg(self, func):
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
        flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
        func_hash = compute_function_hash(flowchart)
        func_addr_set = set([addr for addr in idautils.FuncItems(func)])
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

    def extract_all(self, keep_thunk=False, keep_unnamed=True, keep_small=False, small_ins_threshold=10,
                    small_graph_threshold=3, keep_large=False, large_ins_threshold=100000,
                    large_graph_threshold=100000, max_length=512, monitor_host='localhost', monitor_port=5000):
        client = HeartbeatClient(monitor_host, monitor_port)
        asm_parser = IDAIns()
        client.connect()
        for func in idautils.Functions():
            if func not in self.addr2name:
                continue
            client.ping(func)
            func_name = self.addr2name[func]
            if not keep_thunk and (idc.get_func_flags(func) & idc.FUNC_THUNK):
                logger.info(f"Skip thunk function {func_name}, as it's a thunk function.")
                continue
            if idc.get_segm_name(func) in ['.plt', 'extern', '.init', '.fini']:
                logger.info(f"Skip function {func_name}, as it's in .plt, extern, .init, .fini.")
                continue
            if func_name.startswith('sub_') and not keep_unnamed:
                logger.info(f"Skip function {func_name}, as it's unnamed.")
                continue

            asm_list = self.get_asm(func)

            if not keep_small and len(asm_list) < small_ins_threshold:
                logger.info(f"Skip function {func_name}, as it's too small. It has {len(asm_list)} instructions,"
                            f" while the threshold is {small_ins_threshold}")
                continue

            if not keep_large and len(asm_list) > large_ins_threshold:
                logger.info(f"Skip function {func_name}, as it's too large. It has {len(asm_list)} instructions,"
                            f" while the threshold is {large_ins_threshold}")
                continue

            # rawbytes_list = self.get_rawbytes(func)
            cfg, func_hash = self.get_cfg(func)

            if not keep_small and cfg.number_of_nodes() < small_graph_threshold:
                logger.info(f"Skip function {func_name}, as it's too small. It has {cfg.number_of_nodes()} nodes,"
                            f" while the threshold is {small_graph_threshold}")
                continue

            if not keep_large and cfg.number_of_nodes() > large_graph_threshold:
                logger.info(f"Skip function {func_name}, as it's too large. It has {cfg.number_of_nodes()} nodes,"
                            f" while the threshold is {large_graph_threshold}")
                continue

            nodes = list(cfg.nodes())
            nodes.sort()
            asm = []
            bb_len = []
            for node in nodes:
                asm.extend(cfg.nodes[node]['asm'])
                bb_len.append((node, len(cfg.nodes[node]['asm'])))
            yield self.addr2name[func], asm_parser.parseFunction(asm, max_length=max_length,
                                                                 bb_length=bb_len), func_hash, cfg.number_of_nodes()
        client.end()


logger = logging.getLogger("GEN-AST")
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(logging.Formatter("%(filename)s : %(message)s"))
logger.handlers[0].setLevel(logging.ERROR)


def init_file_handler(filename):
    global logger
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s : %(message)s"))
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--database", default="default.sqlite", type=str, help="path to database")
    ap.add_argument("-l", "--logfile", default=None, type=str, help="path to log file")
    ap.add_argument("--original-file", required=True, type=str, help="path to original binary")
    ap.add_argument("--keep-thunk", action="store_true", default=False, required=False, help='keep thunk functions')
    ap.add_argument("--keep-unnamed", action="store_true", default=False, required=False, help='keep thunk functions')
    ap.add_argument("--keep-small", action='store_true', default=False, required=False, help='keep small functions')
    ap.add_argument("--small-ins-threshold", type=int, default=10, required=False,
                    help='small function instruction threshold')
    ap.add_argument("--small-graph-threshold", type=int, default=3, required=False,
                    help='small function graph threshold')
    ap.add_argument("--keep-large", action='store_true', default=False, required=False, help='keep large functions')
    ap.add_argument("--large-ins-threshold", type=int, default=10000, required=False,
                    help='large function instruction threshold')
    ap.add_argument("--large-graph-threshold", type=int, default=500, required=False,
                    help='large function graph threshold')
    ap.add_argument("--max-length", type=int, default=512, required=False,
                    help='The maximum length of the token list of a function, if'
                         'a function is longer than this, it will be truncated.')
    ap.add_argument("--arch", type=str, required=True, help='The architecture of the binary.')
    ap.add_argument("--monitor-host", required=True, type=str, help="the host of the heartbeat server.")
    ap.add_argument("--monitor-port", required=True, type=int, help="the port of the heartbeat server.")
    args = ap.parse_args(idc.ARGV[1:])

    if args.logfile:
        init_file_handler(args.logfile)

    unstriped_file = args.original_file
    binary_data = BinaryData(unstriped_file)
    saved_path = args.database
    idc.auto_wait()

    cfg_list = []
    extracted_functions = binary_data.extract_all(keep_thunk=args.keep_thunk, keep_unnamed=args.keep_unnamed,
                                                  keep_small=args.keep_small,
                                                  small_ins_threshold=args.small_ins_threshold,
                                                  small_graph_threshold=args.small_graph_threshold,
                                                  keep_large=args.keep_large,
                                                  large_ins_threshold=args.large_ins_threshold,
                                                  large_graph_threshold=args.large_graph_threshold,
                                                  max_length=args.max_length, monitor_host=args.monitor_host,
                                                  monitor_port=args.monitor_port)

    for func_name, token_id_list, func_hash, node_num in extracted_functions:
        cfg_list.append(JTransSeq(func_name, arch=args.arch, token_id=token_id_list, func_hash=func_hash, node_num=node_num))

    with open(saved_path, 'wb') as f:
        pickle.dump(cfg_list, f)
    idc.qexit(0)
