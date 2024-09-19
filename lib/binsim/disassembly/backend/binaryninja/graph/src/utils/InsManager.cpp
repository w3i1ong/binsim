#include "InsManager.h"

bool BinSim::RegOperand::operator==(const BinSim::RegOperand &other) const {
    return reg == other.reg;
}

bool BinSim::RegOperand::operator<(const BinSim::RegOperand &other) const {
    return reg < other.reg;
}

bool BinSim::SpecialTokenOperand::operator==(const BinSim::SpecialTokenOperand &other) const {
    return token == other.token;
}

bool BinSim::SpecialTokenOperand::operator<(const BinSim::SpecialTokenOperand &other) const {
    return token < other.token;
}

bool BinSim::ImmOperand::operator==(const BinSim::ImmOperand &other) const {
    return imm == other.imm;
}

bool BinSim::ImmOperand::operator<(const BinSim::ImmOperand &other) const {
    return imm < other.imm;
}

bool BinSim::X86MemOperand::operator==(const BinSim::X86MemOperand &other) const {
    return base == other.base && index == other.index && scale == other.scale && disp == other.disp;
}

bool BinSim::X86MemOperand::operator<(const BinSim::X86MemOperand &other) const {
    if (base != other.base) return base < other.base;
    if (index != other.index) return index < other.index;
    if (scale != other.scale) return scale < other.scale;
    return disp < other.disp;
}

bool BinSim::ARMMemOperand::operator==(const BinSim::ARMMemOperand &other) const {
    return base == other.base && index == other.index && shift_type == other.shift_type && shift_value == other.shift_value && disp == other.disp;
}

bool BinSim::ARMMemOperand::operator<(const BinSim::ARMMemOperand &other) const {
    if (base != other.base) return base < other.base;
    if (index != other.index) return index < other.index;
    if (shift_type != other.shift_type) return shift_type < other.shift_type;
    if (shift_value != other.shift_value) return shift_value < other.shift_value;
    return disp < other.disp;
}

bool BinSim::MIPSMemOperand::operator==(const BinSim::MIPSMemOperand &other) const {
    return base == other.base && index == other.index && disp == other.disp;
}

bool BinSim::MIPSMemOperand::operator<(const BinSim::MIPSMemOperand &other) const {
    if (base != other.base) return base < other.base;
    if (index != other.index) return index < other.index;
    return disp < other.disp;
}

bool BinSim::ARMRegisterShiftOperand::operator==(const BinSim::ARMRegisterShiftOperand &other) const {
    return reg == other.reg && shift_type == other.shift_type && shift_value == other.shift_value;
}

bool BinSim::ARMRegisterShiftOperand::operator<(const BinSim::ARMRegisterShiftOperand &other) const {
    if (reg != other.reg) return reg < other.reg;
    if (shift_type != other.shift_type) return shift_type < other.shift_type;
    return shift_value < other.shift_value;
}

bool BinSim::ARMImmShiftOperand::operator==(const BinSim::ARMImmShiftOperand &other) const {
    return imm == other.imm && shift_type == other.shift_type && shift_value == other.shift_value;
}

bool BinSim::ARMImmShiftOperand::operator<(const BinSim::ARMImmShiftOperand &other) const {
    if (imm != other.imm) return imm < other.imm;
    if (shift_type != other.shift_type) return shift_type < other.shift_type;
    return shift_value < other.shift_value;
}

bool BinSim::ARMVectorRegisterIndex::operator==(const BinSim::ARMVectorRegisterIndex &other) const {
    return reg == other.reg && index == other.index && vec_type == other.vec_type;
}

bool BinSim::ARMVectorRegisterIndex::operator<(const BinSim::ARMVectorRegisterIndex &other) const {
    if (reg != other.reg) return reg < other.reg;
    if (index != other.index) return index < other.index;
    return vec_type < other.vec_type;
}

bool BinSim::RegisterList::operator==(const BinSim::RegisterList &other) const {
    return reg_list_idx == other.reg_list_idx;
}

bool BinSim::RegisterList::operator<(const BinSim::RegisterList &other) const {
    return reg_list_idx < other.reg_list_idx;
}

bool BinSim::Operand::operator==(const BinSim::Operand &other) const {
    if (type != other.type) return false;
    switch (type) {
        case REG:
            return reg == other.reg;
        case SPECIAL_TOKEN:
            return special_token == other.special_token;
        case IMM:
            return imm == other.imm;
        case X86_MEM:
            return x86_mem == other.x86_mem;
        case ARM_MEM:
            return arm_mem == other.arm_mem;
        case MIPS_MEM:
            return mips_mem == other.mips_mem;
        case ARM_REG_SHIFT:
            return arm_reg_shift == other.arm_reg_shift;
        case ARM_IMM_SHIFT:
            return arm_imm_shift == other.arm_imm_shift;
        case ARM_VEC_REG_INDEX:
            return arm_vec_reg == other.arm_vec_reg;
        case REG_LIST:
            return reg_list == other.reg_list;
        default:
            return false;
    }
}

bool BinSim::Operand::operator<(const BinSim::Operand &other) const {
    if (type != other.type) return type < other.type;
    switch (type) {
        case REG:
            return reg < other.reg;
        case SPECIAL_TOKEN:
            return special_token < other.special_token;
        case IMM:
            return imm < other.imm;
        case X86_MEM:
            return x86_mem < other.x86_mem;
        case ARM_MEM:
            return arm_mem < other.arm_mem;
        case MIPS_MEM:
            return mips_mem < other.mips_mem;
        case ARM_REG_SHIFT:
            return arm_reg_shift < other.arm_reg_shift;
        case ARM_IMM_SHIFT:
            return arm_imm_shift < other.arm_imm_shift;
        case ARM_VEC_REG_INDEX:
            return arm_vec_reg < other.arm_vec_reg;
        case REG_LIST:
            return reg_list < other.reg_list;
        default:
            return false;
    }
}

bool BinSim::Instruction::operator==(const BinSim::Instruction &other) const {
    return opcode == other.opcode && operands == other.operands;
}

bool BinSim::Instruction::operator<(const BinSim::Instruction &other) const {
    if (opcode != other.opcode) return opcode < other.opcode;
    return operands < other.operands;
}

int BinSim::InstructionManager::add_instruction(const BinSim::Instruction &ins) {
    if (ins2id.find(ins) == ins2id.end()) {
        ins2id[ins] = (int)ins2id.size();
        for (const auto &opnd: ins.operands) {
            add_operand(opnd);
        }
    }
    return ins2id[ins];
}

int BinSim::InstructionManager::add_operand(const BinSim::Operand &opnd) {
    switch (opnd.type) {
        case REG:
            if (reg2id.find(opnd) == reg2id.end()) {
                reg2id[opnd] = (int)reg2id.size();
            }
            return reg2id[opnd];
        case SPECIAL_TOKEN:
            if (special2id.find(opnd) == special2id.end()) {
                special2id[opnd] = (int)special2id.size();
            }
            return special2id[opnd];
        case IMM:
            if (imm2id.find(opnd) == imm2id.end()) {
                imm2id[opnd] = (int)imm2id.size();
            }
            return imm2id[opnd];

        case X86_MEM:
            if (x86mem2id.find(opnd) == x86mem2id.end()) {
                x86mem2id[opnd] = (int)x86mem2id.size();
            }
            return x86mem2id[opnd];
        case ARM_MEM:
            if (armmem2id.find(opnd) == armmem2id.end()) {
                armmem2id[opnd] = (int)armmem2id.size();
            }
            return armmem2id[opnd];
        case MIPS_MEM:
            if (mipsmem2id.find(opnd) == mipsmem2id.end()) {
                mipsmem2id[opnd] = (int)mipsmem2id.size();
            }
            return mipsmem2id[opnd];

        case ARM_REG_SHIFT:
            if (armregshift2id.find(opnd) == armregshift2id.end()) {
                armregshift2id[opnd] = (int)armregshift2id.size();
            }
            return armregshift2id[opnd];
        case ARM_IMM_SHIFT:
            if (armimmshift2id.find(opnd) == armimmshift2id.end()) {
                armimmshift2id[opnd] = (int)armimmshift2id.size();
            }
            return armimmshift2id[opnd];
        case ARM_VEC_REG_INDEX:
            if (armvecreg2id.find(opnd) == armvecreg2id.end()) {
                armvecreg2id[opnd] = (int)armvecreg2id.size();
            }
            return armvecreg2id[opnd];
        case REG_LIST:
            return opnd.reg_list.reg_list_idx;
        default:
            binsim_assert(0, "Invalid operand type %d.\n", opnd.type);
    }
    return -1;
}

vector<int> BinSim::InstructionManager::add_basic_block(const vector<uint16_t> &insns,
                                                             const vector<float>& imm_table) {
    int tmp, opnd_num, opcode;
    size_t index;
    vector<Operand> operands;
    vector<int> basic_block_ids;
    for(index = 0; index < insns.size(); ){
        binsim_assert(index >= 0 && index < insns.size(), "Invalid index %zu, len=%zu.\n", index, insns.size());
        opnd_num = insns[index];
        opcode = insns[index+1];
        index += 2;
        operands.resize(0);
        while (opnd_num--){
            auto type = (INS_OPND_TYPE)insns[index++];
            binsim_assert(type <= 10 && type > 0, "Meet an invalid operand type %d at %zu, len=%zu.\n", insns[index], index, insns.size());
            Operand opnd{};
            opnd.type = type;
            switch (type) {
                case REG:
                    opnd.reg.reg = insns[index];
                    index += 1;
                    break;
                case SPECIAL_TOKEN:
                    opnd.special_token.token = insns[index];
                    index += 1;
                    break;
                case IMM:
                    tmp = insns[index];
                    index += 1;
                    binsim_assert(tmp < (int)imm_table.size(), "Invalid immediate value index %d", tmp);
                    opnd.imm.imm = imm_table[tmp];
                    break;
                case X86_MEM:
                    opnd.x86_mem.base = insns[index];
                    opnd.x86_mem.index = insns[index+1];
                    opnd.x86_mem.scale = insns[index+2];
                    tmp = insns[index+3];
                    index += 4;
                    binsim_assert(tmp < (int)imm_table.size(), "Invalid immediate value index %d", tmp);
                    opnd.x86_mem.disp = imm_table[tmp];
                    break;
                case ARM_MEM:
                    opnd.arm_mem.base = insns[index];
                    opnd.arm_mem.index = insns[index+1];
                    opnd.arm_mem.shift_type = insns[index+2];
                    opnd.arm_mem.shift_value = insns[index+3];
                    tmp = insns[index+4];
                    index += 5;
                    binsim_assert(tmp < (int)imm_table.size(), "Invalid immediate value index %d", tmp);
                    opnd.arm_mem.disp = imm_table[tmp];
                    break;
                case MIPS_MEM:
                    opnd.mips_mem.base = insns[index];
                    opnd.mips_mem.index = insns[index+1];
                    tmp = insns[index+2];
                    index += 3;
                    binsim_assert(tmp < (int)imm_table.size(), "Invalid immediate value index %d", tmp);
                    opnd.mips_mem.disp = imm_table[tmp];
                    break;
                case ARM_REG_SHIFT:
                    opnd.arm_reg_shift.reg = insns[index];
                    opnd.arm_reg_shift.shift_type = insns[index+1];
                    opnd.arm_reg_shift.shift_value = insns[index+2];
                    index += 3;
                    break;
                case ARM_IMM_SHIFT:
                    tmp = insns[index];
                    opnd.arm_imm_shift.imm = imm_table[tmp];
                    opnd.arm_imm_shift.shift_type = insns[index+1];
                    opnd.arm_imm_shift.shift_value = insns[index+2];
                    index += 3;
                    break;
                case ARM_VEC_REG_INDEX:
                    opnd.arm_vec_reg.reg = insns[index];
                    opnd.arm_vec_reg.index = insns[index+1];
                    opnd.arm_vec_reg.vec_type = insns[index+2];
                    index += 3;
                    break;
                case REG_LIST:
                    int size = insns[index];
                    index += 1;
                    vector<uint16_t > regs;
                    regs.reserve(size);
                    for(int i = 0; i < size; i++){
                        regs.push_back(insns[index]);
                        index += 1;
                    }
                    if(reglist2id.find(regs) == reglist2id.end()){
                        reglist2id[regs] = (int)reglist2id.size();
                    }
                    opnd.reg_list.reg_list_idx = reglist2id[regs];
                    break;
            }
            operands.push_back(opnd);
        }
        Instruction ins;
        ins.opcode = opcode;
        ins.operands = operands;
        int ins_id = add_instruction(ins);
        basic_block_ids.push_back(ins_id);
    }
    return basic_block_ids;
}

map<string, py::array> BinSim::InstructionManager::get_instruction_features() const{
    map<string, py::array> features;
    vector<int> tmp;
    vector<float> tmp_float;
    int reg_op_base = 0;
    int special_token_base, imm_base, x86_mem_base, arm_mem_base, mips_mem_base,
            arm_reg_shift_base, arm_imm_shift_base, arm_vec_reg_base, reg_list_base;
    // opcode
    if(!ins2id.empty()){
        tmp.resize(ins2id.size());
        for(auto& ins: ins2id){
            binsim_assert(ins.second < (int)ins2id.size(), "Invalid instruction id %d, size=%zu.\n", ins.second, ins2id.size());
            tmp[ins.second] = ins.first.opcode;
        }
        features["mnemonic"] = vector_to_array(tmp);
    }
    // operands
    // register
    if(!reg2id.empty()){
        tmp.resize(reg2id.size());
        for(auto& reg: reg2id){
            binsim_assert(reg.second < (int)reg2id.size(), "Invalid register id %d, size=%zu.\n", reg.second, reg2id.size());
            tmp[reg.second] = reg.first.reg.reg;
        }
        features["register"] = vector_to_array(tmp);
    }
    // special token
    special_token_base = reg_op_base + (int)reg2id.size();
    if(!special2id.empty()){
        tmp.resize(special2id.size());
        for(auto& special: special2id){
            binsim_assert(special.second < (int)special2id.size(), "Invalid special token id %d, size=%zu.\n", special.second, special2id.size());
            tmp[special.second] = special.first.special_token.token;
        }
        features["token"] = vector_to_array(tmp);
    }
    // immediate
    imm_base = special_token_base + (int) special2id.size();
    if(!imm2id.empty()) {
        tmp_float.resize(imm2id.size());
        for (auto &imm: imm2id) {
            binsim_assert(imm.second < (int) imm2id.size(), "Invalid immediate id %d, size=%zu.\n", imm.second,
                          imm2id.size());
            tmp_float[imm.second] = imm.first.imm.imm;
        }
        features["immediate"] = vector_to_array(tmp_float);
    }
    // x86 memory
    x86_mem_base = imm_base + (int) imm2id.size();
    if(!x86mem2id.empty()) {
        tmp.resize(x86mem2id.size() * 3);
        tmp_float.resize(x86mem2id.size());
        for (auto &x86mem: x86mem2id) {
            binsim_assert(x86mem.second < (int) x86mem2id.size(), "Invalid x86 memory id %d, size=%zu.\n",
                          x86mem.second, x86mem2id.size());
            tmp[x86mem.second * 3] = x86mem.first.x86_mem.base;
            tmp[x86mem.second * 3 + 1] = x86mem.first.x86_mem.index;
            tmp[x86mem.second * 3 + 2] = x86mem.first.x86_mem.scale;
            tmp_float[x86mem.second] = x86mem.first.x86_mem.disp;
        }
        features["x86_memory.tokens"] = vector_to_array(tmp);
        features["x86_memory.disp"] = vector_to_array(tmp_float);
    }
    // arm memory
    arm_mem_base = x86_mem_base + (int)x86mem2id.size();
    if(!armmem2id.empty()) {
        tmp.resize(armmem2id.size()*4);
        tmp_float.resize(armmem2id.size());
        for(auto& armmem: armmem2id){
            binsim_assert(armmem.second < (int)armmem2id.size(), "Invalid arm memory id %d, size=%zu.\n", armmem.second, armmem2id.size());
            tmp[armmem.second*4] = armmem.first.arm_mem.base;
            tmp[armmem.second*4+1] = armmem.first.arm_mem.index;
            tmp[armmem.second*4+2] = armmem.first.arm_mem.shift_type;
            tmp[armmem.second*4+3] = armmem.first.arm_mem.shift_value;
            tmp_float[armmem.second] = armmem.first.arm_mem.disp;
        }
        features["arm_memory.tokens"] = vector_to_array(tmp);
        features["arm_memory.disp"] = vector_to_array(tmp_float);
    }

    mips_mem_base = arm_mem_base + (int) armmem2id.size();
    if(!mipsmem2id.empty()) {
        // mips memory
        tmp.resize(mipsmem2id.size() * 2);
        tmp_float.resize(mipsmem2id.size());
        for (auto &mipsmem: mipsmem2id) {
            binsim_assert(mipsmem.second < (int) mipsmem2id.size(), "Invalid mips memory id %d, size=%zu.\n",
                          mipsmem.second, mipsmem2id.size());
            tmp[mipsmem.second * 2] = mipsmem.first.mips_mem.base;
            tmp[mipsmem.second * 2 + 1] = mipsmem.first.mips_mem.index;
            tmp_float[mipsmem.second] = mipsmem.first.mips_mem.disp;
        }
        features["mips_memory.tokens"] = vector_to_array(tmp);
        features["mips_memory.disp"] = vector_to_array(tmp_float);
    }
    // arm register shift
    arm_reg_shift_base = mips_mem_base + (int) mipsmem2id.size();
    if(!armregshift2id.empty()) {
        tmp.resize(armregshift2id.size() * 3);
        for (auto &armregshift: armregshift2id) {
            binsim_assert(armregshift.second < (int) armregshift2id.size(),
                          "Invalid arm register shift id %d, size=%zu.\n", armregshift.second, armregshift2id.size());
            tmp[armregshift.second * 3] = armregshift.first.arm_reg_shift.reg;
            tmp[armregshift.second * 3 + 1] = armregshift.first.arm_reg_shift.shift_type;
            tmp[armregshift.second * 3 + 2] = armregshift.first.arm_reg_shift.shift_value;
        }
        features["arm_reg_shift"] = vector_to_array(tmp);
    }
    // arm imm shift
    arm_imm_shift_base = arm_reg_shift_base + (int)armregshift2id.size();
    if(!armimmshift2id.empty()) {
        tmp.resize(armimmshift2id.size()*2);
        tmp_float.resize(armimmshift2id.size());
        for(auto& armimmshift: armimmshift2id){
            binsim_assert(armimmshift.second < (int)armimmshift2id.size(), "Invalid arm imm shift id %d, size=%zu.\n", armimmshift.second, armimmshift2id.size());
            tmp[armimmshift.second*2+1] = armimmshift.first.arm_imm_shift.shift_value;
            tmp[armimmshift.second*2] = armimmshift.first.arm_imm_shift.shift_type;
            tmp_float[armimmshift.second] = armimmshift.first.arm_imm_shift.imm;
        }
        features["arm_imm_shift.tokens"] = vector_to_array(tmp);
        features["arm_imm_shift.imm"] = vector_to_array(tmp_float);
    }
    arm_vec_reg_base = arm_imm_shift_base + (int) armimmshift2id.size();
    // arm vector register index
    if(!armvecreg2id.empty()) {
        tmp.resize(armvecreg2id.size() * 3);
        for (auto &armvecreg: armvecreg2id) {
            binsim_assert(armvecreg.second < (int) armvecreg2id.size(),
                          "Invalid arm vector register index id %d, size=%zu.\n", armvecreg.second,
                          armvecreg2id.size());
            tmp[armvecreg.second * 3] = armvecreg.first.arm_vec_reg.reg;
            tmp[armvecreg.second * 3 + 1] = armvecreg.first.arm_vec_reg.index;
            tmp[armvecreg.second * 3 + 2] = armvecreg.first.arm_vec_reg.vec_type;
        }
        features["arm_vec_reg"] = vector_to_array(tmp);
    }
    reg_list_base = arm_vec_reg_base + (int)armvecreg2id.size();
    // register list
    if(!reglist2id.empty()) {
        vector<int> register_list;
        vector<int> index;
        for(auto& reglist: reglist2id){
            for(auto& reg: reglist.first){
                register_list.push_back(reg);
                index.push_back(reglist.second);
            }
        }
        features["reg_list"] = vector_to_array(register_list);
        features["reg_list.index"] = vector_to_array(index);
    }
    // ins opnd
    for(int i = 0; i < 10; i++){
        vector<int> ins_opnd;
        vector<int> ins_idx;
        for(auto& ins: ins2id){
            if((int)ins.first.operands.size() <= i)
                continue;
            ins_idx.push_back(ins.second);
            auto opnd = ins.first.operands[i];
            switch (opnd.type) {
                case REG:
                    ins_opnd.push_back(reg2id.find(opnd)->second);
                    break;
                case SPECIAL_TOKEN:
                    ins_opnd.push_back(special2id.find(opnd)->second + special_token_base);
                    break;
                case IMM:
                    ins_opnd.push_back(imm2id.find(opnd)->second + imm_base);
                    break;
                case X86_MEM:
                    ins_opnd.push_back(x86mem2id.find(opnd)->second + x86_mem_base);
                    break;
                case ARM_MEM:
                    ins_opnd.push_back(armmem2id.find(opnd)->second + arm_mem_base);
                    break;
                case MIPS_MEM:
                    ins_opnd.push_back(mipsmem2id.find(opnd)->second + mips_mem_base);
                    break;
                case ARM_REG_SHIFT:
                    ins_opnd.push_back(armregshift2id.find(opnd)->second + arm_reg_shift_base);
                    break;
                case ARM_IMM_SHIFT:
                    ins_opnd.push_back(armimmshift2id.find(opnd)->second + arm_imm_shift_base);
                    break;
                case ARM_VEC_REG_INDEX:
                    ins_opnd.push_back(armvecreg2id.find(opnd)->second + arm_vec_reg_base);
                    break;
                case REG_LIST:
                    ins_opnd.push_back(opnd.reg_list.reg_list_idx + reg_list_base);
                    break;
            }
        }
        if(ins_opnd.empty())
            break;
        features["ins_operand.op_idx_" + to_string(i)] = vector_to_array(ins_opnd);
        features["ins_operand.ins_idx_" + to_string(i)] = vector_to_array(ins_idx);
    }
    return features;
}
