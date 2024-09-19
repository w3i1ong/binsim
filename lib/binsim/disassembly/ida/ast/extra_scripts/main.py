# encoding=utf-8
"""
Python2.7
IDAPython script running with Hexray plugin !!!
usage: idat -Sast_generator.py binary|binary.idb
Extracting the asts of all functions in the binary file and save the function information along with ast to the database file
"""
import time

import idautils
from idc import *
from idaapi import *
from idautils import *
import idc
import idaapi
import ida_pro
import logging, os, sys
import pickle
from binsim.disassembly.ida.ast import Tree
from hashlib import sha256
from binsim.disassembly.utils import BinaryBase
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
                elif operand_type == 1:
                    operators.append('[REG]')
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
                    operand = print_operand(ea, op_idx)
                    if operand is not None:
                        operators.append(print_operand(ea, op_idx))
            assert mnem is not None, "mnemonic is None, please check the instruction at %s" % hex(ea)
            ea = idc.next_head(ea)
    return sha256("".join(operators).encode()).hexdigest()


cur_dir = os.path.dirname(os.path.abspath(__file__))
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


logger.info("using IDA Version {}".format(IDA_SDK_VERSION))

if IDA_SDK_VERSION < 700:
    # IDAPro 6.x To 7.x (https://www.hex-rays.com/products/ida/support/ida74_idapython_no_bc695_porting_guide.shtml)
    logger.info(f"Only support IDA 7.x, but you are using IDA {IsADirectoryError}, please upgrade your IDA")


# ---- prepare environment
def wait_for_analysis_to_finish():
    """
    :return:
    """
    logger.info('[+] waiting for analysis to finish...')
    auto_wait()
    logger.info('[+] analysis finished')


def load_plugin_decompiler():
    """
    load the hexray plugins
    :return: success or not
    """
    is_ida64 = get_idb_path().endswith(".i64")
    if not is_ida64:
        idaapi.load_plugin("hexrays")
        idaapi.load_plugin("hexarm")
    else:
        idaapi.load_plugin("hexx64")
    if not idaapi.init_hexrays_plugin():
        logger.error('[+] decompiler plugins load failed. IDAdb: %s' % get_input_file_path())
        idc.Exit(0)


def get_arch_name():
    arch_info = idaapi.get_inf_structure()
    arch = arch_info.procName
    bits = 64 if arch_info.is_64bit() else 32
    # todo: normalize arch name
    return f"{arch}:{bits}"


wait_for_analysis_to_finish()
load_plugin_decompiler()

# -----------------------------------

# --------------------------
spliter = "************"


class Visitor(idaapi.ctree_visitor_t):
    # preorder traversal tree
    def __init__(self, cfunc):
        idaapi.ctree_visitor_t.__init__(self, idaapi.CV_FAST | idaapi.CV_INSNS)
        self.cfunc = cfunc
        self._op_type_list = []
        self._op_name_list = []
        self._tree_struction_list = []
        self._id_list = []
        self._statement_num = 0
        self._callee_set = set()
        self.root = None  # root node of tree

    # Generate the sub tree
    def GenerateAST(self, ins):
        self._statement_num += 1
        AST = Tree()
        try:
            logger.info("[insn] op  %s" % (ins.opname))
            AST.op = ins.op
            AST.opname = ins.opname

            if ins.op == idaapi.cit_block:
                self.dump_block(ins.ea, ins.cblock, AST)
            elif ins.op == idaapi.cit_expr:
                AST.add_child(self.dump_expr(ins.cexpr))

            elif ins.op == idaapi.cit_if:
                logger.info("[if]" + spliter)
                cif = ins.details
                cexpr = cif.expr
                ithen = cif.ithen
                ielse = cif.ielse

                AST.add_child(self.dump_expr(cexpr))
                if ithen:
                    AST.add_child(self.GenerateAST(ithen))
                if ielse:
                    AST.add_child(self.GenerateAST(ielse))

            elif ins.op == idaapi.cit_while:
                cwhile = ins.details
                self.dump_while(cwhile, AST)

            elif ins.op == idaapi.cit_return:
                creturn = ins.details
                AST.add_child(self.dump_return(creturn))

            elif ins.op == idaapi.cit_for:
                logger.info('[for]' + spliter)
                cfor = ins.details
                AST.add_child(self.dump_expr(cfor.init))
                AST.add_child(self.dump_expr(cfor.step))
                AST.add_child(self.dump_expr(cfor.expr))
                AST.add_child(self.GenerateAST(cfor.body))
            elif ins.op == idaapi.cit_switch:
                logger.info('[switch]' + spliter)
                cswitch = ins.details
                cexpr = cswitch.expr
                ccases = cswitch.cases  # Switch cases: values and instructions.
                cnumber = cswitch.mvnf  # Maximal switch value and number format.
                AST.add_child(self.dump_expr(cexpr))
                self.dump_ccases(ccases, AST)
            elif ins.op == idaapi.cit_do:
                logger.info('[do]' + spliter)
                cdo = ins.details
                cbody = cdo.body
                cwhile = cdo.expr
                AST.add_child(self.GenerateAST(cbody))
                AST.add_child(self.dump_expr(cwhile))
            elif ins.op == idaapi.cit_break or ins.op == idaapi.cit_continue:
                pass
            elif ins.op == idaapi.cit_goto:
                pass
            else:
                logger.error(f'Meet an unknown op type {ins.opname}, skipped.')

        except Exception as e:
            logger.warning(f"Meet error when generating AST: {e}")

        return AST

    def visit_insn(self, ins):
        # pre-order visit ctree Generate new AST
        # ins maybe None , why ?

        if not ins:
            return 1
        # l.info("[AST] address and op %s %s" % (hex(ins.ea), ins.opname))
        self.root = self.GenerateAST(ins)
        logger.info(self.root)
        return 1

    def dump_return(self, creturn):
        """
        return an expression?
        """
        return self.dump_expr(creturn.expr)

    def dump_while(self, cwhile, parent):
        """
        visit while statement
        return:
            condition: expression tuple
            body : block
        """
        expr = cwhile.expr
        parent.add_child(self.dump_expr(expr))
        whilebody = None
        body = cwhile.body
        if body:
            parent.add_child(self.GenerateAST(body))

    def dump_ccases(self, ccases, parent_node):
        """
        :param ccases:
        :return: return a list of cases
        """
        for ccase in ccases:
            AST = Tree()
            AST.opname = 'case'
            AST.op = ccase.op
            logger.info('case opname %s, op %d' % (ccase.opname, ccase.op))
            value = 0  # default
            size = ccase.size()  # List of case values. if empty, then 'default' case , ： 'acquire', 'append', 'disown', 'next', 'own
            if size > 0:
                value = ccase.value(0)
            AST.value = value
            block = self.dump_block(ccase.ea, ccase.cblock, AST)
            parent_node.add_child(AST)

    def dump_expr(self, cexpr):
        """
        l.info the expression
        :return: AST with two nodes op and oprand : op Types.NODETYPE.OPTYPE, oprand : list[]
        """
        # l.info "dumping expression %x" % (cexpr.ea)

        oprand = []  # a list of Tree()
        logger.info("[expr] op %s" % cexpr.opname)

        if cexpr.op == idaapi.cot_call:
            # oprand = args
            # get the function call arguments
            self._get_callee(cexpr.ea)
            logger.info('[call]' + spliter)
            args = cexpr.a
            for arg in args:
                oprand.append(self.dump_expr(arg))
        elif cexpr.op == idaapi.cot_idx:
            logger.info('[idx]' + spliter)
            oprand.append(self.dump_expr(cexpr.x))
            oprand.append(self.dump_expr(cexpr.y))

        elif cexpr.op == idaapi.cot_memptr:
            logger.info('[memptr]' + spliter)
            AST = Tree()
            AST.op = idaapi.cot_num  # consider the mem size pointed by memptr
            AST.value = cexpr.ptrsize
            AST.opname = "value"
            oprand.append(AST)
            # oprand.append(cexpr.m) # cexpr.m : member offset
            # oprand.append(cexpr.ptrsize)
        elif cexpr.op == idaapi.cot_memref:

            offset = Tree()
            offset.op = idaapi.cot_num
            offset.opname = "offset"
            offset.addr = cexpr.ea
            offset.value = cexpr.m
            oprand.append(offset)

        elif cexpr.op == idaapi.cot_num:
            logger.info('[num]' + str(cexpr.n._value))
            AST = Tree()
            AST.op = idaapi.cot_num  # consider the mem size pointed by memptr
            AST.value = cexpr.n._value
            AST.opname = "value"
            oprand.append(AST)

        elif cexpr.op == idaapi.cot_var:

            var = cexpr.v
            entry_ea = var.mba.entry_ea
            idx = var.idx
            ltree = Tree()
            ltree.op = idaapi.cot_memptr
            ltree.addr = cexpr.ea
            ltree.opname = 'entry_ea'
            ltree.value = entry_ea
            oprand.append(ltree)
            rtree = Tree()
            rtree.value = idx
            rtree.op = idaapi.cot_num
            rtree.addr = cexpr.ea
            rtree.opname = 'idx'
            oprand.append(rtree)

        elif cexpr.op == idaapi.cot_str:
            # string constant
            logger.info('[str]' + cexpr.string)
            AST = Tree()
            AST.opname = "string"
            AST.op = cexpr.op
            AST.value = cexpr.string
            oprand.append(AST)

        elif cexpr.op == idaapi.cot_obj:
            logger.info('[cot_obj]' + hex(cexpr.obj_ea))
            # oprand.append(cexpr.obj_ea)
            # Many strings are defined as 'obj'
            # I wonder if 'obj' still points to other types of data?
            # notice that the address of 'obj' is not in .text segment
            if get_segm_name(getseg(cexpr.obj_ea)) not in ['.text']:
                AST = Tree()
                AST.opname = "string"
                AST.op = cexpr.op
                AST.value = get_strlit_contents(cexpr.obj_ea, -1, 0)
                oprand.append(AST)

        elif idaapi.cot_fdiv >= cexpr.op >= idaapi.cot_comma:
            # All binocular operators
            oprand.append(self.dump_expr(cexpr.x))
            oprand.append(self.dump_expr(cexpr.y))

        elif idaapi.cot_fneg <= cexpr.op <= idaapi.cot_call:
            # All unary operators
            logger.info('[single]' + spliter)
            oprand.append(self.dump_expr(cexpr.x))
        else:
            logger.error(f'Meet an unknown op type {cexpr.opname}, skipped.')
        AST = Tree()
        AST.opname = cexpr.opname
        AST.op = cexpr.op
        for tree in oprand:
            AST.add_child(tree)
        return AST

    def dump_block(self, ea, b, parent):
        """
        :param ea: block address
        :param b:  block_structure
        :param parent: parent node
        :return:
        """
        # iterate over all block instructions
        for ins in b:
            if ins:
                parent.add_child(self.GenerateAST(ins))

    def get_caller(self):
        call_addrs = list(idautils.CodeRefsTo(self.cfunc.entry_ea, 0))
        return len(set(call_addrs))

    def get_callee(self):
        return len(self._callee_set)

    def _get_callee(self, ea):
        '''
        :param ea:  where the call instruction points to
        :return: None
        '''
        logger.info('analyse addr %s callee' % hex(ea))
        addrs = list(idautils.CodeRefsFrom(ea, 0))
        for addr in addrs:
            if addr == get_func_attr(addr, 0):
                self._callee_set.add(addr)


class AstGenerator(BinaryBase):

    def __init__(self, original_file):
        super().__init__(original_file)
        self.fix_up()
        self.bin_file_path = get_input_file_path()  # path to binary
        self.file_name = get_root_filename()  # name of binary
        # get process info
        self.bits, self.arch, self.endian = self._get_process_info()
        self.function_info_list = list()
        # Save the information of all functions, of which ast class is saved using pick.dump.
        # Each function is saved with a tuple (func_name. func_addr, ast_pick_dump, pseudocode, callee, caller)

    def fix_up(self):
        for addr in self.addr2name:
            # incase some functions' instructions are not recognized by IDA
            idc.create_insn(addr)
            idc.add_func(addr)

    def _get_process_info(self):
        """
        :return: 32 or 64 bit, arch, endian
        """
        info = idaapi.get_inf_structure()
        bits = 32
        if info.is_64bit():
            bits = 64
        try:
            is_be = info.is_be()
        except:
            is_be = info.mf
        endian = "big" if is_be else "little"
        return bits, info.procName, endian

    def progreeBar(self, i):
        sys.stdout.write('\r%d%% [%s]' % (int(i), "#" * i))
        sys.stdout.flush()

    def run(self,
            fn,
            special_name="",
            keep_thunk=False,
            keep_unnamed=False,
            keep_small=False,
            small_ins_threshold=10,
            small_graph_threshold=3,
            keep_large=False,
            large_ins_threshold=1000,
            large_graph_threshold=700,
            heartbeat_host="localhost",
            heartbeat_port=0):
        """
        :param fn: a function to handle the functions in binary
        :param special_name: specific function name while other functions are ignored
        :param keep_thunk: whether to keep thunk functions
        :param keep_unnamed: whether to keep unnamed functions
        :param keep_small: whether to keep small functions
        :param small_ins_threshold: the threshold of the number of instructions in a function. If keep_small is False,
            functions with less than small_ins_threshold instructions will be discarded.
        :param small_graph_threshold: the threshold of the number of basic blocks in a function. If keep_small is False,
            functions with less than small_graph_threshold basic blocks will be discarded.
        :param keep_large: whether to keep large functions
        :param large_ins_threshold: the threshold of the number of instructions in a function. If keep_large is False,
            functions with more than large_ins_threshold instructions will be discarded.
        :param large_graph_threshold: the threshold of the number of basic blocks in a function. If keep_large is False,
            functions with more than large_graph_threshold basic blocks will be discarded.
        :return:
        """
        if heartbeat_host is not None:
            client = HeartbeatClient(host=heartbeat_host, port=heartbeat_port)
            function_to_skip = client.connect()
        else:
            client = None
            function_to_skip = set()
        if special_name != "":
            logger.info("specific function name %s" % special_name)
        for i in range(0, get_func_qty()):
            func = getn_func(i)
            address = func.start_ea
            if address in function_to_skip:
                continue
            if client is not None:
                client.ping(address)

            if not keep_unnamed and address not in self.addr2name:
                continue
            func_name = self.addr2name[address]
            if not keep_unnamed and func_name.startswith("sub_"):
                continue

            self.progreeBar(int((i * 1.0) / get_func_qty() * 100))
            segname = get_segm_name(getseg(address))
            if segname[1:3] not in ["OA", "OM", "te", "_t"]:
                continue

            # skip thunk functions
            if not keep_thunk and (func.flags & idc.FUNC_THUNK):
                continue

            flow_charts = FlowChart(func)
            fuc_hash = compute_function_hash(flow_charts)
            bb_num = flow_charts.size
            ins_num = len(list(FuncItems(func.start_ea)))

            # skip small functions
            if not keep_small and (ins_num <= small_ins_threshold or bb_num <= small_graph_threshold):
                continue

            # skip large functions
            if not keep_large and (ins_num > large_ins_threshold or bb_num > large_graph_threshold):
                continue

            if len(special_name) > 0 and special_name != func_name:
                continue
            try:
                ast_tree, callee_num, caller_num = fn(func)
                self.function_info_list.append(ast_tree.as_code_ast(func_name, arch=get_arch_name(),
                                                                    func_hash=fuc_hash, callee_num=callee_num,
                                                                    node_num=bb_num))
            except Exception as e:
                logger.error("Meet error while trying to save ast: %s" % e)
        if client is not None:
            client.end()

    def save_to(self, filename):
        """
        :param filename: the path to save the database
        :return:
        """
        logger.debug(f"Save {len(self.function_info_list)} functions into {filename}")
        with open(filename, "wb") as f:
            pickle.dump(self.function_info_list, f)
        return

    @staticmethod
    def get_info_of_func(func):
        """
        :param func:
        :return:
        """
        try:
            cfunc = idaapi.decompile(func.start_ea)
            vis = Visitor(cfunc)
            vis.apply_to(cfunc.body, None)
            return vis.root, vis.get_callee(), vis.get_caller()
        except Exception as e:
            logger.error(f"Meet error while trying to decompile function @ {hex(func.start_ea)}: {str(e)}")


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--function", default="", help="extract the specific function info")
    ap.add_argument("-d", "--database", default="default.sqlite", type=str, help="path to database")
    ap.add_argument("-l", "--logfile", default=None, type=str, help="path to log file")
    ap.add_argument("--keep-thunk", action="store_true", default=False, required=False, help='keep thunk functions')
    ap.add_argument("--keep-unnamed", action="store_true", default=False, required=False, help='keep thunk functions')
    ap.add_argument("--keep-small", action='store_true', default=False, required=False, help='keep small functions')
    ap.add_argument("--small-ins-threshold", type=int, default=10, required=False,
                    help='small function instruction threshold')
    ap.add_argument("--small-graph-threshold", type=int, default=10, required=False,
                    help='small function graph threshold')
    ap.add_argument("--keep-large", action='store_true', default=False, required=False, help='keep large functions')
    ap.add_argument("--large-ins-threshold", type=int, default=1000, required=False,
                    help='large function instruction threshold')
    ap.add_argument("--large-graph-threshold", type=int, default=1000, required=False,
                    help='large function graph threshold')
    ap.add_argument("--original-file", required=True, type=str, help="path to original binary")
    ap.add_argument("--monitor-host", required=False, default=None, type=str, help="the host of the heartbeat server.")
    ap.add_argument("--monitor-port", required=False, default=None, type=int, help="the port of the heartbeat server.")
    args = ap.parse_args(idc.ARGV[1:])
    if args.logfile:
        init_file_handler(args.logfile)
    astg = AstGenerator(args.original_file)
    astg.run(astg.get_info_of_func,
             special_name=args.function,
             keep_thunk=args.keep_thunk,
             keep_unnamed=args.keep_unnamed,
             keep_small=args.keep_small,
             small_ins_threshold=args.small_ins_threshold,
             small_graph_threshold=args.small_graph_threshold,
             keep_large=args.keep_large,
             large_ins_threshold=args.large_ins_threshold,
             large_graph_threshold=args.large_graph_threshold,
             heartbeat_port=args.monitor_port,
             heartbeat_host=args.monitor_host)
    astg.save_to(args.database)
    ida_pro.qexit(0)
