#ifndef DATAFLOW_INS_MANAGER_H
#define DATAFLOW_INS_MANAGER_H
#include <map>
#include <vector>
#include <cstdint>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "array.h"
using namespace std;
namespace py = pybind11;

namespace BinSim{
    enum INS_OPND_TYPE{
        X86_MEM = 1,
        ARM_MEM = 2,
        MIPS_MEM = 3,
        REG_LIST = 4,
        REG = 5,
        IMM = 6,
        SPECIAL_TOKEN = 7,
        ARM_REG_SHIFT = 8,
        ARM_IMM_SHIFT = 9,
        ARM_VEC_REG_INDEX = 10
    };

    struct RegOperand{
        uint16_t reg;
        bool operator == (const RegOperand& other) const;
        bool operator < (const RegOperand& other) const;
    };

    struct SpecialTokenOperand{
        uint16_t token;
        bool operator == (const SpecialTokenOperand& other) const;
        bool operator < (const SpecialTokenOperand& other) const;
    };

    struct ImmOperand{
        float imm;
        bool operator == (const ImmOperand& other) const;
        bool operator < (const ImmOperand& other) const;
    };

    struct X86MemOperand{
        uint16_t base;
        uint16_t index;
        uint16_t scale;
        float disp;
        bool operator == (const X86MemOperand& other) const;
        bool operator < (const X86MemOperand& other) const;
    };

    struct ARMMemOperand{
        uint16_t base;
        uint16_t index;
        uint16_t shift_type;
        uint16_t shift_value;
        float disp;
        bool operator == (const ARMMemOperand& other) const;
        bool operator < (const ARMMemOperand& other) const;
    };

    struct MIPSMemOperand{
        uint16_t base;
        uint16_t index;
        float disp;
        bool operator == (const MIPSMemOperand& other) const;
        bool operator < (const MIPSMemOperand& other) const;
    };

    struct ARMRegisterShiftOperand{
        uint16_t reg;
        uint16_t shift_type;
        uint16_t shift_value;
        bool operator == (const ARMRegisterShiftOperand& other) const;
        bool operator < (const ARMRegisterShiftOperand& other) const;
    };

    struct ARMImmShiftOperand{
        float imm;
        uint16_t shift_type;
        uint16_t shift_value;
        bool operator == (const ARMImmShiftOperand& other) const;
        bool operator < (const ARMImmShiftOperand& other) const;
    };

    struct ARMVectorRegisterIndex{
        uint16_t reg;
        uint16_t index;
        uint16_t vec_type;
        bool operator == (const ARMVectorRegisterIndex& other) const;
        bool operator < (const ARMVectorRegisterIndex& other) const;
    };


    struct RegisterList{
        int reg_list_idx;
        bool operator == (const RegisterList& other) const;
        bool operator < (const RegisterList& other) const;
    };

    struct Operand{
        INS_OPND_TYPE type;
        union{
            RegOperand reg;
            SpecialTokenOperand special_token;
            ImmOperand imm;
            X86MemOperand x86_mem;
            ARMMemOperand arm_mem;
            MIPSMemOperand mips_mem;
            ARMRegisterShiftOperand arm_reg_shift;
            ARMImmShiftOperand arm_imm_shift;
            ARMVectorRegisterIndex arm_vec_reg;
            RegisterList reg_list;
        };
        bool operator==(const Operand& other) const;
        bool operator<(const Operand& other) const;
    };

    struct Instruction{
        uint16_t opcode;
        vector<Operand> operands;
        bool operator==(const Instruction& other) const;
        bool operator<(const Instruction& other) const;
    };

    class InstructionManager{
    private:
        map<Instruction, int> ins2id;
        map<Operand, int> reg2id;
        map<Operand, int> special2id;
        map<Operand, int> imm2id;
        map<Operand, int> x86mem2id;
        map<Operand, int> armmem2id;
        map<Operand, int> mipsmem2id;
        map<Operand, int> armregshift2id;
        map<Operand, int> armimmshift2id;
        map<Operand, int> armvecreg2id;
        map<vector<uint16_t>, int> reglist2id;
    public:
        int add_instruction(const Instruction& ins);
        int add_operand(const Operand& opnd);
        vector<int> add_basic_block(const vector<uint16_t>& insns, const vector<float>& imm_table);
        map<string, py::array> get_instruction_features() const;
    };
}

#endif
