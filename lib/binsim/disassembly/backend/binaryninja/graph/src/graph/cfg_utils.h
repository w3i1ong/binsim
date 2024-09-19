#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <string>
#include "../utils/SimpleSerializer.h"
#include "../utils/array.h"
#include "../utils/utils.h"
#include "../utils/algorithm.h"
#ifndef DATAUTILS_CFG_UTILS_H
#define DATAUTILS_CFG_UTILS_H

enum DATA_TYPE{
    BYTECODE = 0,
    ACFG = 1,
    TOKEN_SEQ = 2,
    TOKEN_CFG = 3,
    JTRANS_SEQ = 4,
    CODE_AST = 5,
    INS_CFG = 6,
    INS_SEQ = 7,
    INS_DAG = 9
};

namespace py = pybind11;
using namespace std;
namespace BinSim{
    py::bytes bytecode_pack_neural_input(uint in_degree, uint out_degree, const std::string& payload);
    tuple<py::array, py::array> bytecode_collate_neural_input(const vector<string>& inputs, int max_byte_num);

    py::bytes acfg_pack_neural_input(uint16_t node_num,
                                     const vector<uint16_t>& src,
                                     const vector<uint16_t>& dst,
                                     const vector<vector<float>>&features);
    tuple<vector<tuple<py::array, py::array>>, py::array, py::array>
            acfg_collate_neural_input(const vector<string>& inputs);

    py::bytes token_seq_pack_neural_input(const vector<int>&features, int max_length=300);
    vector<py::array> token_seq_collate_neural_input(const vector<string>& inputs, int chunks, int max_length=300);

    py::bytes token_cfg_pack_neural_input(uint16_t node_num,
                                          const vector<uint16_t>& src,
                                          const vector<uint16_t>& dst,
                                          const vector<vector<int>>&features);
    pair<vector<pair<py::array, py::array>>, vector<py::array>>
            token_cfg_collate_neural_input(const vector<string>& inputs, int chunks=-1);

    py::bytes code_ast_pack_neural_input(const vector<int>& src, const vector<int>& dst,
                                         int node_num, int callee_num,
                                         const vector<int>&features);
    tuple<vector<pair<py::array, py::array>>, vector<py::array>> code_ast_collate_neural_input(const vector<string>& inputs);

    py::bytes jtrans_seq_pack_neural_input(const vector<int>&features);
    vector<py::array> jtrans_seq_cfg_collate_neural_input(const vector<string>& inputs);

    py::bytes ins_dag_pack_neural_input(uint16_t node_num, const vector<uint16_t>& src, const vector<uint16_t>& dst,
                                        const vector<uint16_t> & node_id,
                                        const vector<vector<uint16_t>>& features, const vector<float>& imm_values);
    tuple<map<string,py::array>,vector<pair<py::array, py::array>>, vector<py::array>>
        ins_dag_collate_neural_input(const vector<string>& inputs, int chunks);

    py::bytes ins_cfg_pack_neural_input(const vector<vector<uint16_t>>& adjList,
                                        const vector<vector<uint16_t>>& features, const vector<float>& imm_values);
    py::bytes ins_cfg_preprocess_neural_input(const string& imm_values);
    tuple<map<string,py::array>,vector<pair<py::array, py::array>>, vector<py::array>>
    ins_cfg_collate_neural_input(const vector<string>& inputs, int chunks);

    py::bytes ins_seq_pack_neural_input(const vector<uint16_t> &features, const vector<float> &imm_values);
    tuple<map<string,py::array>,vector<py::array>>
        ins_seq_collate_neural_input(const vector<string>& inputs, int chunks, bool pack=false);
}

#endif //DATAUTILS_CFG_UTILS_H
