import os, logging
from .codeast import Tree, CodeAST
from binsim.disassembly.core import CFGNormalizerBase
from binsim.disassembly.backend.ida.graph.utils import compute_function_hash

cur_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("GEN-AST")
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(logging.Formatter("%(filename)s : %(message)s"))
logger.handlers[0].setLevel(logging.ERROR)

class CodeASTNormalizer(CFGNormalizerBase):
    def __init__(self, arch):
        super().__init__(arch)

    def __call__(self, func)->CodeAST:
        import idc, idaapi
        from idautils import FuncItems
        flow_charts = idaapi.FlowChart(func)
        fuc_hash = compute_function_hash(flow_charts)
        func_name = idc.get_func_name(func.start_ea)
        func_info = self.get_info_of_func(func)
        if func_info is None:
            return
        ast_tree, callee_num, caller_num = func_info
        ins_num = len(list(FuncItems(func.start_ea)))
        return ast_tree.as_code_ast(func_name, arch=self.arch, func_hash=fuc_hash,
                                    callee_num=callee_num, node_num=flow_charts.size,
                                    ins_num=ins_num)

    @staticmethod
    def get_info_of_func(func):
        """
        :param func:
        :return:
        """
        import idaapi
        try:
            cfunc = idaapi.decompile(func.start_ea)
            vis = Visitor(cfunc)
            vis.apply_to(cfunc.body, None)
            return vis.root, vis.get_callee(), vis.get_caller()
        except Exception as e:
            logger.error(f"Meet error while trying to decompile function @ {hex(func.start_ea)}: {str(e)}")


try:
    import idaapi, idautils
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
                from ida_segment import  get_segm_name
                from idaapi import getseg
                from idc import get_strlit_contents
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
            from idc import get_func_attr
            '''
            :param ea:  where the call instruction points to
            :return: None
            '''
            logger.info('analyse addr %s callee' % hex(ea))
            addrs = list(idautils.CodeRefsFrom(ea, 0))
            for addr in addrs:
                if addr == get_func_attr(addr, 0):
                    self._callee_set.add(addr)
except ImportError:
    pass
