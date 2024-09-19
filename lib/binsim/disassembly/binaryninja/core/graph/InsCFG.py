import logging
import pickle

from binaryninja import Function, BasicBlock, InstructionTextTokenType as TokenType
from typing import Dict, List, Set, Tuple, Union
from ..utils import compute_function_hash
from .TokenCFG import TokenCFG, TokenCFGMode, TokenCFGDataForm, CFGNormalizerBase
from .instruction import BSInstruction, REGListOperand, \
    X86MemOperand, REGOperand, IMMOperand, SpecialTokenOperand, ARMMemOperand, MIPSMemOperand, \
    ARMRegisterShiftOperand, ARMImmShiftOperand, ARMVectorRegisterIndex
from binaryninja.architecture import InstructionTextToken
from dgl import DGLGraph
from binsim.utils import init_logger
logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)


class SpecialTokens:
    NoREG = "[NoREG]"
    NoShift = "[NoShift]"
    RelCode = "[Rel-C]"
    UknMem = "[UknMem]"
    GlobalMem = "[Addr]"
    UknToken = "[UlnToken]"
    RelFunc = "[Rel-F]"
    PossibleAddr = "[MaybeAddr]"
    ShiftValMap = {}

    @staticmethod
    def shift_value(n):
        if n in SpecialTokens.ShiftValMap:
            return SpecialTokens.ShiftValMap[n]
        if n != 0:
            result = f'[ShiftValue-{n}]'
        else:
            result = '[NoShift]'
        SpecialTokens.ShiftValMap[n] = result
        return result

    @staticmethod
    def register_index(idx):
        return f"[RegIndex-{idx}]"

    def __init__(self):
        raise NotImplementedError("SpecialTokens cannot be instantiated.")


class InsCFG(TokenCFG):
    def __init__(self,
                 function_name: str,
                 adj_list: Dict[int, List],
                 features: Dict[int, List[List[str]]],
                 func_hash: str,
                 func_arch: str,
                 form=TokenCFGDataForm.InsStrGraph,
                 entry_points=None):
        """
        Build a token CFG from the given adjacency list and features.
        :param function_name: The name of the function.
        :param adj_list: The adjacency list of the CFG.
        :param features: A list, in which each element is a list of tokens.
        :param func_hash: The hash of the function, used to remove duplicate functions.
        :param func_arch: The architecture of the function.
        :param form: The form of the input.
            If set to 'sequence', a sequence of tokens will be returned.
            If set to 'graph', a DGLGraph and two tensors will be returned. The first tensor is a block_num * max_token_num
            tensor, which represents the instruction sequences in each basic block. The second tensor is a block_num tensor,
            which represents the length of instruction sequence in each basic block.
        """
        super().__init__(function_name,
                         adj_list=adj_list,
                         features=features,
                         func_hash=func_hash,
                         func_arch=func_arch,
                         form=form,
                         entry_points=entry_points,
                         node_num=len(adj_list))

    def unique_tokens(self) -> Set[Union[str, int]]:
        """
        Get the all unique tokens in the CFG.
        :return: A set of unique tokens.
        """
        result = set()
        for addr, basic_block in self.features.items():
            for inst in basic_block:
                result.add(inst.mnemonic)
                for operand in inst.operands:
                    if isinstance(operand, REGOperand):
                        result.add(operand.reg)
                    elif isinstance(operand, SpecialTokenOperand):
                        result.add(operand.token)
                    elif isinstance(operand, X86MemOperand):
                        result.update([operand.base, operand.index])
                    elif isinstance(operand, ARMMemOperand):
                        result.update([operand.base, operand.index, operand.shift_type, operand.shift_value])
                    elif isinstance(operand, ARMRegisterShiftOperand):
                        result.update([operand.shift_type, operand.register, operand.value])
                    elif isinstance(operand, ARMImmShiftOperand):
                        result.add([operand.shift_type, operand.value])
                    elif isinstance(operand, ARMVectorRegisterIndex):
                        result.update([operand.register, operand.type, operand.index])
                    elif isinstance(operand, REGListOperand):
                        result.update(operand.regs)
                    elif isinstance(operand, MIPSMemOperand):
                        result.update([operand.base, operand.index])
                    else:
                        raise ValueError(f"Meet an unknown operand type: {type(operand)}.")
        return result

    def replace_tokens(self, token2id: Dict[str, int], record_unseen=True, update_token2id=False)->Set|None:
        """
        Replace tokens in the CFG with token ids.
        :param token2id: A dictionary, which maps tokens to token ids.
        :return:
        """
        if self._mode == TokenCFGMode.ID:
            return None
        unseen_tokens = set()
        total_token_num, total_unseen_num = 0, 0
        def get_token_id(token: str):
            nonlocal total_token_num, total_unseen_num
            total_token_num += 1
            token_id = token2id.get(token, 0)
            if token_id == 0:
                total_unseen_num += 1
                if record_unseen:
                    unseen_tokens.add(token)
                if update_token2id:
                    token_id = len(token2id)
                    token2id[token] = token_id
            return token_id

        for address, basic_block in self.features.items():
            for instruction in basic_block:
                instruction.mnemonic = get_token_id(instruction.mnemonic)
                for idx, operand in enumerate(instruction.operands):
                    if isinstance(operand, REGOperand):
                        instruction.operands[idx] = REGOperand(reg=get_token_id(operand.reg))
                    elif isinstance(operand, IMMOperand):
                        continue
                    elif isinstance(operand, SpecialTokenOperand):
                        instruction.operands[idx] = SpecialTokenOperand(token=get_token_id(operand.token))
                    elif isinstance(operand, X86MemOperand):
                        instruction.operands[idx] = X86MemOperand(base=get_token_id(operand.base),
                                                                  index=get_token_id(operand.index),
                                                                  scale=operand.scale,
                                                                  disp=operand.disp,
                                                                  bits=operand.bits)
                    elif isinstance(operand, ARMMemOperand):
                        instruction.operands[idx] = ARMMemOperand(base=get_token_id(operand.base),
                                                                  index=get_token_id(operand.index),
                                                                  shift_type=get_token_id(operand.shift_type),
                                                                  shift_value=get_token_id(operand.shift_value),
                                                                  disp=operand.disp)
                    elif isinstance(operand, ARMRegisterShiftOperand):
                        instruction.operands[idx] = ARMRegisterShiftOperand(register=get_token_id(operand.register),
                                                                            shift_type=get_token_id(operand.shift_type),
                                                                            value=get_token_id(operand.value))
                    elif isinstance(operand, ARMImmShiftOperand):
                        instruction.operands[idx] = ARMImmShiftOperand(imm=operand.imm,
                                                                       shift_type=get_token_id(operand.shift_type),
                                                                       value=get_token_id(operand.value))
                    elif isinstance(operand, ARMVectorRegisterIndex):
                        instruction.operands[idx] = ARMVectorRegisterIndex(register=get_token_id(operand.register),
                                                                           type=get_token_id(operand.type),
                                                                           index=get_token_id(operand.index))
                    elif isinstance(operand, REGListOperand):
                        instruction.operands[idx] = REGListOperand(regs=tuple([get_token_id(reg) for reg in operand.regs]))
                    elif isinstance(operand, MIPSMemOperand):
                        instruction.operands[idx] = MIPSMemOperand(base=get_token_id(operand.base),
                                                                   index=get_token_id(operand.index),
                                                                   disp=operand.disp,
                                                                   bits=operand.bits)
                    else:
                        raise ValueError(f"Meet an unknown operand type: {type(operand)}.")

        self._mode = TokenCFGMode.ID
        return unseen_tokens, total_token_num, total_unseen_num

    def as_token_list(self) -> List[str]:
        raise NotImplementedError("Token list is not supported in InsCFG.")

    def random_walk(self, *args, **kwargs) -> List[List[str]]:
        raise NotImplementedError("Random walk is not supported in InsCFG.")

    def __repr__(self):
        return f'<InsCFG::{self.name}' \
               f'(node_num={len(self._features)})>'

    def as_neural_input(self, expand_time=None, as_seq=False, max_seq_length=None) -> Union[
        Tuple[DGLGraph, List, List], List]:
        """
        Transform current graph to the input of neural network.
        :param expand_time: int or None
            Specify the expansion time of loop expansion.
            If None, the original graph will be returned;
            If int, its value should be a non-negative integer.
        :param as_seq: bool
            If set to True, expand_time will be ignored, the function will be represented as a instruction sequence.
            It set to False, the function will be represented as a graph.
        :return:
        """
        if not as_seq:
            cfg, id2addr = self.as_dgl_graph(expand_time=expand_time)
            if max_seq_length is None:
                basic_blocks = [self.features[addr] for idx, addr in id2addr.items()]
            else:
                basic_blocks = [self.features[addr][:max_seq_length] for idx, addr in id2addr.items()]
            length = [len(bb) for bb in basic_blocks]
            assert min(length) != 0, f"Meet a basic block without any instruction: {self.name}"
            return cfg, basic_blocks, length
        else:
            assert expand_time is None, "The graph will be represented as an instruction sequence, but the expand-time " \
                                        "is given."
            result = []
            for addr in self.basic_block_address:
                result.extend(self.features[addr])
                if len(result) > max_seq_length:
                    break
            return result[:max_seq_length]


class InsCFGNormalizer(CFGNormalizerBase):
    Token2Bits = {'byte': 8, 'word': 16, 'dword': 32, 'qword': 64, 'tword': 80, 'oword': 128, 'yword': 256,
                  'zword': 512, 'xmmword': 128, 'ymmword': 256, 'zmmword': 512
                  }
    ARM_SHIFT_TOKENS = {'lsl', 'lsr', 'asr', 'ror', 'msl', 'rrx'}
    ARM_CONDITION_TOKEN = {"eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge",
                           "lt", "gt", "le", "al"}
    ARM_EXTEND_TOKENS = {'uxtb', 'uxth', 'uxtw', 'uxtx', 'sxtb', 'sxth', 'sxtw', 'sxtx', 'sxt'}

    def __init__(self, arch, token2idx: dict[str, int] | str = None):
        super().__init__(arch=arch)
        if isinstance(token2idx, str):
            with open(token2idx, 'rb') as f:
                token2idx = pickle.load(f)
        self._ins2idx = token2idx
        self._extract_tokens = getattr(self, f'extract_bb_tokens_{self.arch.value}')

    def __call__(self, function: Function) -> TokenCFG:
        adj_list, entry_points = self.extract_adj_list(function)
        token_sequences = self._extract_tokens(function)
        # some basic blocks contain no instruction and no successors, just remove them
        for basic_block in function.basic_blocks:
            if basic_block.instruction_count == 0 and len(basic_block.outgoing_edges) == 0:
                token_sequences.pop(basic_block.start)
                adj_list.pop(basic_block.start)
                for edge in basic_block.incoming_edges:
                    src_addr = edge.source.start
                    adj_list[src_addr].remove(basic_block.start)
        # end of the check

        cfg = InsCFG(function.name, adj_list, token_sequences,
                     func_hash=compute_function_hash(function),
                     func_arch=self.arch,
                     entry_points=entry_points)
        if self._ins2idx is not None:
            cfg.replace_tokens(self._ins2idx, record_unseen=False, update_token2id=False)
        return cfg

    @staticmethod
    def extract_bb_tokens_x64(function: Function) -> Dict[int, List[BSInstruction]]:
        """
        Extract basic blocks for x64 functions.
        :param function: function extracted by Binary Ninja
        :return:
        """
        results = dict()
        for basic_block in function:
            results[basic_block.start] = InsCFGNormalizer._extract_bb_tokens_x86(basic_block, bits=64)
        return results

    @staticmethod
    def extract_bb_tokens_x86(function: Function) -> Dict[int, List[BSInstruction]]:
        results = dict()
        for basic_block in function:
            results[basic_block.start] = InsCFGNormalizer._extract_bb_tokens_x86(basic_block, bits=32)
        return results

    @staticmethod
    def extract_bb_tokens_arm32(function: Function) -> Dict[int, List[BSInstruction]]:
        results = dict()
        for basic_block in function:
            results[basic_block.start] = InsCFGNormalizer._extract_bb_tokens_arm(basic_block, bits=32)
        return results

    @staticmethod
    def extract_bb_tokens_arm64(function: Function) -> Dict[int, List[BSInstruction]]:
        results = dict()
        for basic_block in function:
            results[basic_block.start] = InsCFGNormalizer._extract_bb_tokens_arm(basic_block, bits=64)
        return results

    @staticmethod
    def extract_bb_tokens_mips32(function: Function) -> Dict[int, List[BSInstruction]]:
        results = dict()
        for basic_block in function:
            results[basic_block.start] = InsCFGNormalizer._extract_bb_tokens_mips(basic_block)
        return results

    @staticmethod
    def extract_bb_tokens_mips64(function: Function) -> Dict[int, List[BSInstruction]]:
        results = dict()
        for basic_block in function:
            results[basic_block.start] = InsCFGNormalizer._extract_bb_tokens_mips(basic_block)
        return results

    @staticmethod
    def _extract_bb_tokens_mips(basic_block: BasicBlock) -> List[BSInstruction]:
        result = []
        for ins, byte_length in basic_block:
            result.append(InsCFGNormalizer._parse_instruction_mips(ins))
        return result

    @staticmethod
    def _parse_instruction_mips(ins) -> BSInstruction:
        # find the operator
        ins_str = ''.join([token.text for token in ins])
        for token in ins:
            token.text = token.text.strip()
        ins = [token for token in ins if len(token.text) or token.type != TokenType.TextToken]
        mnemonic = ins[0].text
        idx, length, operands = 1, len(ins), []
        # split the operands
        while idx < length:
            idx, operand = InsCFGNormalizer._next_operand(ins, idx)
            try:
                operands.append(InsCFGNormalizer._parse_operand_mips(operand))
            except:
                raise ValueError(f"Meet error while processing operand: {operand} in instruction: {ins_str}. "
                                 f"Token types: {[token.type.name for token in operand]}")

        return BSInstruction(mnemonic, operands)

    @staticmethod
    def _parse_operand_mips(tokens):
        # remove spaces in the operand
        tokens = [token for token in tokens if not token.text.isspace()]
        if len(tokens) == 1:
            token = tokens[0]
            if token.type == TokenType.RegisterToken:
                return REGOperand(reg=token.text)
            elif token.type == TokenType.IntegerToken:
                return IMMOperand(imm=InsCFGNormalizer.normalize_imm(token.value))
            elif token.type == TokenType.PossibleAddressToken:
                return SpecialTokenOperand(token=SpecialTokens.PossibleAddr)
            elif token.type == TokenType.CodeRelativeAddressToken:
                return SpecialTokenOperand(token=SpecialTokens.RelCode)
            elif token.type == TokenType.BeginMemoryOperandToken:
                return SpecialTokenOperand(token=SpecialTokens.UknMem)
            else:
                raise ValueError(f"Meet error while processing a single-token operand: {token}. "
                                 f"Unknown token type: {format(token.type.name)}")
        else:
            first_token = tokens[0]
            match first_token.type:
                case TokenType.BeginMemoryOperandToken:
                    if len(tokens) == 6:
                        _, offset, left_bracket, base, right_bracket, _ = tokens
                        assert base.type == TokenType.RegisterToken, f"Meet error while processing memory operand: {tokens}."
                        if offset.type == TokenType.IntegerToken:
                            return MIPSMemOperand(base=base.text, index=SpecialTokens.NoREG,
                                                  disp=InsCFGNormalizer.normalize_imm(offset.value), bits=32)
                        elif offset.type == TokenType.RegisterToken:
                            return MIPSMemOperand(base=base.text, index=offset.text, disp=0, bits=32)
                        else:
                            raise RuntimeError(f"Meet error while processing memory operand: {tokens}.")
                    elif len(tokens) == 5:
                        _, left_bracket, base, right_bracket, _ = tokens
                        assert base.type == TokenType.RegisterToken
                        return MIPSMemOperand(base=base.text, index=SpecialTokens.NoREG, disp=0, bits=32)
                    elif len(tokens) == 2:
                        return SpecialTokenOperand(SpecialTokens.GlobalMem)
                    elif len(tokens) == 1:
                        return SpecialTokenOperand(SpecialTokens.UknMem)
                    else:
                        raise ValueError(
                            f"Meet error while processing memory operand: {tokens}. Expected 5 or 6 tokens,"
                            f" but got {len(tokens)} tokens.")
                case _:
                    raise ValueError(f"Meet an operand with unknown type: {first_token.type.name}.")

    @staticmethod
    def _extract_bb_tokens_arm(basic_block: BasicBlock, bits=32) -> List[BSInstruction]:
        result = []
        for ins, byte_length in basic_block:
            result.append(InsCFGNormalizer._parse_instruction_arm(ins, bits=bits))
        return result

    def _extract_bb_tokens_arm64(self, basic_block: BasicBlock) -> List[BSInstruction]:
        return self._extract_bb_tokens_arm(basic_block, bits=64)

    @staticmethod
    def _transform_arm_instruction(ins) -> Tuple[InstructionTextToken, List]:
        mnemonic = ins[0]
        length, operands = len(ins), []
        assert length == 1 or ins[1].text.isspace(), "There are multiple mnemonics in one instruction."

        tokens = []
        # remove space tokens
        for token_index, token in enumerate(ins):
            token.text = token.text.strip()
            # skip empty TextTokens
            if token.type == TokenType.TextToken and len(token.text) == 0:
                continue

            match token.type:
                case TokenType.EndMemoryOperandToken:
                    rightBracket, comma, leftBrace = InstructionTextToken(TokenType.EndMemoryOperandToken, ']'), \
                        InstructionTextToken(TokenType.OperandSeparatorToken, ','), \
                        InstructionTextToken(TokenType.TextToken, '{')
                    match token.text:
                        case ']':
                            tokens.append(token)
                        case '],':
                            tokens.extend([rightBracket, comma])
                        case '], {':
                            tokens.extend([rightBracket, comma, leftBrace])
                        case '], #':
                            tokens.extend([rightBracket, comma, InstructionTextToken(TokenType.TextToken, '#')])
                        case ']!':
                            token.text = ']'
                            tokens.append(token)
                        case _:
                            raise ValueError(
                                f"Meet an unknown token: {token.text} while processing instruction: {ins}.")
                case TokenType.TextToken:
                    if len(token.text) <= 2 or token.text.isalnum():
                        tokens.append(token)
                        continue
                    if token.text.endswith('#'):
                        first_token = token.text[:-1].strip()
                        if InsCFGNormalizer.is_arm_shift(first_token):
                            assert tokens[-1].type == TokenType.OperandSeparatorToken
                            tokens[-1].type = TokenType.TextToken
                            tokens.append(InstructionTextToken(TokenType.TextToken,first_token))
                            token.text = '#'
                            tokens.append(token)
                        elif first_token == ',':
                            tokens.append(InstructionTextToken(TokenType.OperandSeparatorToken, ','))
                            token.text = '#'
                            tokens.append(token)
                        else:
                            raise ValueError(f"Meed an unknown token:{token.text} while processing instruction: {ins}")
                        continue
                    tokens.append(token)
                case _:
                    tokens.append(token)

        ins = [token for token in tokens if len(token.text)]
        idx, length = 1, len(ins)
        operands = []

        # split the operands
        while idx < length:
            idx, operand = InsCFGNormalizer._next_operand(ins, idx)
            operands.append(operand)
        return mnemonic, operands

    @staticmethod
    def _parse_instruction_arm(ins, bits=32) -> BSInstruction:
        ins_text = ''.join([token.text for token in ins])
        mnemonic, operands = InsCFGNormalizer._transform_arm_instruction(ins)
        # find the mnemonic
        opnds = []
        for opnd in operands:
            try:
                opnds.append(InsCFGNormalizer._parse_operand_arm(opnd, bits=bits))
            except Exception as e:
                logger.error(f"Meet error while processing operand: {opnd} in instruction: \"{ins_text}\", skipped.")
                logger.error(f"Meet error while processing operand: {opnd} in instruction: \"{ins_text}\", skipped.")
        return BSInstruction(mnemonic.text, opnds)

    @staticmethod
    def _parse_operand_arm(operand, bits=32):
        # discard prefix '-'
        if operand[0].text == '-' and operand[1].type == TokenType.RegisterToken:
            operand = operand[1:]
        token_number = len(operand)

        # 1. single token operand.
        if token_number == 1:
            token = operand[0]
            match token.type:
                case TokenType.RegisterToken:
                    return REGOperand(reg=token.text)
                case TokenType.IntegerToken | TokenType.FloatingPointToken:
                    return IMMOperand(imm=InsCFGNormalizer.normalize_imm(token.value))
                case TokenType.PossibleAddressToken:
                    return SpecialTokenOperand(token=SpecialTokens.PossibleAddr)
                case TokenType.CodeRelativeAddressToken:
                    return SpecialTokenOperand(token=SpecialTokens.RelCode)
                case TokenType.TextToken:
                    if token.text not in InsCFGNormalizer.ARM_CONDITION_TOKEN:
                        logger.debug(f"Meet a single text token operands: {token.text}")
                    return SpecialTokenOperand(token=token.text)
                case _:
                    raise ValueError(f"Meet error while processing operand: {token.text}. "
                                     f"Unknown token type: {format(token.type)}")
        # 2. immediate number token
        if token_number == 2 and operand[0].text == '#':
            match operand[1].type:
                case TokenType.IntegerToken | TokenType.FloatingPointToken:
                    return IMMOperand(imm=InsCFGNormalizer.normalize_imm(operand[1].value))
                case TokenType.PossibleAddressToken:
                    return SpecialTokenOperand(token=SpecialTokens.PossibleAddr)
                case _:
                    raise ValueError(f"Meet error while processing operand: {operand}. "
                                     f"Expected an Integer or FloatingPoint, but got {operand[1].type.name}")
        # memory operands
        if operand[0].type == TokenType.BeginMemoryOperandToken:
            # remove postfix !, as we currently don't use it.
            if operand[-1].text == '!':
                operand.pop()
                token_number -= 1
            # current operand is a memory operand in the form of [...]
            if operand[-1].type == TokenType.EndMemoryOperandToken:
                base, offset, index, shift, shift_op = None, 0, None, 0, \
                    SpecialTokens.NoShift
                idx = 1
                while idx < token_number - 1:
                    cur_token = operand[idx]
                    idx += 1
                    # skip ','
                    if cur_token.text == ',':
                        continue
                    # base or index register
                    if cur_token.type == TokenType.RegisterToken:
                        if base is None:
                            base = cur_token.text
                        else:
                            index = cur_token.text
                        continue
                    # offset
                    elif cur_token.text == '#' or cur_token.text == ', #':
                        assert operand[idx].type == TokenType.IntegerToken
                        offset = operand[idx].value
                        idx += 1
                        continue
                    # shift or extend
                    elif cur_token.type == TokenType.TextToken:
                        if InsCFGNormalizer.is_arm_shift(cur_token.text) or InsCFGNormalizer.is_arm_extend(
                                cur_token.text):
                            if index is None:
                                index, base = base, index
                            shift_op = cur_token.text
                            if operand[idx].text == '#':
                                shift = operand[idx + 1].value
                                idx += 2
                            elif operand[idx].type == TokenType.RegisterToken:
                                raise ValueError(f"Meet error while processing operand: {operand}, "
                                                 f"the shift value is a register.")
                            elif operand[idx].type != TokenType.EndMemoryOperandToken:
                                raise ValueError(f"Meet error while processing operand: {operand}, "
                                                 f"expected a register or an immediate value or a closing bracket, "
                                                 f"but got {operand[idx].text}")
                            continue
                        elif cur_token.text == '-':
                            continue
                        elif cur_token.text == ':0x':
                            return SpecialTokenOperand(token=SpecialTokens.UknMem)
                        else:
                            raise ValueError(
                                f"Meet unknown token {cur_token.text} while processing operand: {operand}, their token types are: "
                                f"{[token.type.name for token in operand]}")
                    else:
                        raise ValueError(
                            f"Meet error while processing operand: {operand}, their token types are: "
                            f"{[token.type.name for token in operand]}")
                if base is None: base = SpecialTokens.NoREG
                if index is None: index = SpecialTokens.NoREG
                return ARMMemOperand(base=base, disp=InsCFGNormalizer.normalize_imm(offset),
                                     index=index, shift_type=shift_op,
                                     shift_value=SpecialTokens.shift_value(shift))
            elif operand[1].type == TokenType.RegisterToken and (operand[2].text == '], #' or operand[2].text == '],'):
                return ARMMemOperand(base=operand[1].text, disp=0, index=SpecialTokens.NoREG,
                                     shift_type=SpecialTokens.NoShift,
                                     shift_value=SpecialTokens.shift_value(0))
            else:
                raise ValueError(f"Meet error while processing operand: {operand}.")
        elif operand[0].text == '{':
            # register list
            start_idx, regs = 1, []
            # sometimes, the register list is empty, why?
            if operand[start_idx].text != '}' and operand[start_idx].text != '#':
                while True:
                    regs.append(operand[start_idx].text)
                    try:
                        if operand[start_idx + 1].text.startswith('.'):
                            start_idx += 1
                        if operand[start_idx + 1].text == '}':
                            break
                        elif operand[start_idx + 1].text == ',':
                            start_idx += 2
                            continue
                        else:
                            raise ValueError(f"Meet an unknown token: {operand[start_idx + 1].text} while processing "
                                             f"register list: {operand}.")
                    except:
                        raise ValueError(f"Meet error while processing register list: {operand}.")
                return REGListOperand(regs=regs)
            elif operand[start_idx].text == '#':
                logger.debug(
                    f"Meet a strange operand: {operand}. Expect a register list, but meet immediate number in it.")
                assert operand[start_idx + 1].type == TokenType.IntegerToken
                assert operand[start_idx + 2].text == '}', f"Meet error while processing register list: {operand}."
                return IMMOperand(imm=InsCFGNormalizer.normalize_imm(operand[start_idx + 1].value))
            raise ValueError(f"Meet error while processing register list: {operand}.")
        elif operand[0].type == TokenType.RegisterToken and len(operand) >= 3:
            register = operand[0].text
            if operand[1].text == ',':
                shift_type, value = operand[2].text, SpecialTokens.shift_value(0)
                if len(operand) > 3:
                    if operand[3].text == '#':
                        assert operand[4].type == TokenType.IntegerToken
                        value = operand[4].value
                        value = SpecialTokens.shift_value(value)
                    elif operand[3].type == TokenType.RegisterToken:
                        value = operand[3].text
                    else:
                        raise ValueError(f"Meet error while processing operand: {operand}, its token types are:"
                                         f"{[token.type.name for token in operand]}")
                return ARMRegisterShiftOperand(shift_type=shift_type, value=value, register=register)
            elif len(operand) == 5 and operand[2].text == '[' and operand[4].text == ']':
                index, register_type = operand[3].text, operand[1].text
                return ARMVectorRegisterIndex(register=register, type=register_type,
                                              index=SpecialTokens.register_index(index))
            elif operand[1].text == '[' and operand[-1].text == ']':
                return REGOperand(reg=register)
            raise ValueError(f"Meet error while processing operand: {operand}, its token types are:"
                             f"{[token.type.name for token in operand]}")

        elif operand[0].text == '#' and len(operand) >= 4 and \
                InsCFGNormalizer.is_arm_shift(operand[3].text):
            imm, shift_type = operand[1].value, operand[3].text
            if operand[4].text == '#':
                assert operand[5].type == TokenType.IntegerToken
                value = SpecialTokens.shift_value(operand[5].value)
            elif operand[4].type == TokenType.RegisterToken:
                value = operand[4].text
            else:
                raise ValueError(f"Meet error while processing operand: {operand}.")
            return ARMImmShiftOperand(shift_type=shift_type, value=value, imm=InsCFGNormalizer.normalize_imm(imm))
        elif len(operand) == 2:
            if operand[0].type == TokenType.RegisterToken:
                return REGOperand(operand[0].text)
            elif operand[0].text == '-' and operand[1].type == TokenType.RegisterToken:
                return REGOperand(operand[1].text)

        raise ValueError(f"Meet error while processing operand: {operand}, unknown token type: {operand[0].type}")

    @staticmethod
    def _extract_bb_tokens_x86_64(basic_block: BasicBlock) -> List[BSInstruction]:
        return InsCFGNormalizer._extract_bb_tokens_x86(basic_block, bits=64)

    @staticmethod
    def _extract_bb_tokens_x86(basic_block: BasicBlock, bits=32) -> List[BSInstruction]:
        """
        Transform Basic Block into BSInstruction List.
        :param basic_block: Basic block extracted by BinaryNinja.
        :param bits:
        :return:
        """
        result = []
        for ins, byte_length in basic_block:
            result.append(InsCFGNormalizer._parse_instruction_x86(ins, bits=bits))
        return result

    @staticmethod
    def _next_operand(tokens, base_idx):
        length = len(tokens)
        # 1. skip spaces
        while base_idx < length and tokens[base_idx].text.isspace():
            base_idx += 1
        if base_idx >= length:
            return base_idx, None

        level, start_idx = 0, base_idx
        while start_idx < length and \
                (tokens[start_idx].type != TokenType.OperandSeparatorToken or level != 0):  # skip Operand Separators
            if tokens[start_idx].text == '{' or tokens[start_idx].text == '[':
                level += 1
            elif tokens[start_idx].text.startswith('}') or tokens[start_idx].text.startswith(']'):
                level -= 1
            start_idx += 1

        return start_idx + 1, tokens[base_idx:start_idx]

    @staticmethod
    def _parse_instruction_x86(ins, bits=32) -> BSInstruction:
        # 1. get mnemonic
        length, mnemonic = len(ins), ins[0].text
        assert length == 1 or ins[1].text.isspace(), "There are multiple mnemonics in one instruction."
        # 2. extract operands
        idx, operands = 2, []
        while idx < length:
            idx, operand = InsCFGNormalizer._next_operand(ins, idx)
            if operand is not None:
                operands.append(InsCFGNormalizer._parse_operand_x86(operand, bits=bits))
        return BSInstruction(mnemonic, operands)

    x86_MEMOP_START_TOKEN = {'byte', 'word', 'dword', 'tword', 'qword', 'xmmword', 'ymmword', 'zmmword', '['}

    @staticmethod
    def _parse_operand_x86(tokens, bits=32):
        operand = []
        for token in tokens:
            token.text = token.text.strip()
            if token.type == TokenType.TextToken:
                if len(token.text) == 0:
                    continue
                elif token.text.startswith('{') and token.text.endswith('}'):
                    continue
            operand.append(token)

        if len(operand) == 1:
            operand = operand[0]
            if operand.type == TokenType.RegisterToken:
                return REGOperand(reg=operand.text)
            elif operand.type == TokenType.IntegerToken:
                return IMMOperand(imm=InsCFGNormalizer.normalize_imm(operand.value))
            elif operand.type == TokenType.PossibleAddressToken:
                # todo: sometimes, the token is not a real address, but an immediate value.
                # we need to distinguish them.
                return SpecialTokenOperand(token=SpecialTokens.PossibleAddr)
            elif operand.type == TokenType.CodeRelativeAddressToken:
                return SpecialTokenOperand(token=SpecialTokens.RelCode)
            elif operand.text.startswith('$+'):  # Relative address inside the function
                return SpecialTokenOperand(token=SpecialTokens.RelFunc)
            else:
                logger.error(f"Meet an unknown single-token operand: {operand}, its type is {operand.type.name}."
                             f"Replace it with [Unknown].")
                return SpecialTokenOperand(token=SpecialTokens.UknToken)
        elif operand[0].text.strip() in InsCFGNormalizer.x86_MEMOP_START_TOKEN:
            if operand[0].text.strip() == '[':
                op_bytes = bits
            else:
                op_bytes = InsCFGNormalizer.get_op_bytes(operand[0].text.strip())
                operand = operand[1:]
            assert operand[0].text.strip() == '[' and operand[-1].text.strip() == ']', \
                "Expect the operand is a memory operand, but it doesn't have a closing quare bracket."

            # memory operands like [0x1000000]
            if len(operand) == 3 and operand[1].type == TokenType.PossibleAddressToken:
                return SpecialTokenOperand(token=SpecialTokens.GlobalMem)

            base = index = special_token = None
            scale, disp = 1, 0

            # we believe the format of memory operand is several registers and several integers separated by arithmetic
            # operator. So we increment the index by 2 each time.
            for i in range(1, len(operand) - 1, 2):
                cur_token = operand[i]
                if cur_token.type == TokenType.RegisterToken:
                    if base is None:
                        base = cur_token.text
                    else:
                        index = cur_token.text
                elif cur_token.type == TokenType.IntegerToken:
                    if operand[i - 1].text.strip() == '-':
                        disp = -cur_token.value
                    else:
                        if operand[i - 1].text.strip() == '*':
                            scale = cur_token.value
                            if index is None:
                                index, base = base, index
                        else:
                            disp = cur_token.value
                elif cur_token.text.strip() == 'rel':
                    special_token = SpecialTokens.GlobalMem
                    assert operand[i + 2].text.strip() == ']', f"Meet an unknown memory operand: {operand}."
                    break
                else:
                    raise ValueError(f"Meet an unknown token {cur_token.text} while processing operand: {operand}, "
                                     f"unknown token type: {format(cur_token.type.name)}")

            if special_token is not None:
                return SpecialTokenOperand(special_token)
            else:
                if base is None: base = SpecialTokens.NoREG
                if index is None: index = SpecialTokens.NoREG
                return X86MemOperand(base=base, index=index, scale=scale, disp=InsCFGNormalizer.normalize_imm(disp),
                                     bits=op_bytes)
        else:
            raise ValueError(f"Meet error while processing operand: {operand}, "
                             f"unknown token type: {[token.type.name for token in operand]}")

    @staticmethod
    def get_op_bytes(s: str):
        return InsCFGNormalizer.Token2Bits[s]

    @staticmethod
    def is_arm_shift(token: str):
        return token in InsCFGNormalizer.ARM_SHIFT_TOKENS

    @staticmethod
    def is_arm_extend(token: str):
        return token in InsCFGNormalizer.ARM_EXTEND_TOKENS

    @staticmethod
    def is_arm_condition(token: str):
        return token in InsCFGNormalizer.ARM_CONDITION_TOKEN

    @staticmethod
    def normalize_imm(n):
        if n < 2 ** 15:
            return n

        if 2 ** 15 <= n < 2 ** 16:
            return -(2 ** 16 - n)

        if n < 2 ** 31:
            return n
        if 2 ** 31 <= n < 2 ** 32:
            return -(2 ** 32 - n)

        if n < 2 ** 63:
            return n

        if 2 ** 63 <= n < 2 ** 64:
            return -(2 ** 64 - n)
        raise ValueError(f"The value {n} is too large to be normalized.")
