#include "graph/cfg_utils.h"
#include <iostream>

PYBIND11_MODULE(datautils, m) {
    m.doc() = "Graph module";

    m.def("bytecode_pack_neural_input", &BinSim::bytecode_pack_neural_input, "Pack neural input for ByteCode.");
    m.def("bytecode_collate_neural_input", &BinSim::bytecode_collate_neural_input, "Collate neural input for ByteCode.");

    m.def("acfg_pack_neural_input", &BinSim::acfg_pack_neural_input, "Pack neural input for ACFG.");
    m.def("acfg_collate_neural_input", &BinSim::acfg_collate_neural_input, "Collate neural input for ACFG.");

    m.def("token_seq_pack_neural_input", &BinSim::token_seq_pack_neural_input);
    m.def("token_seq_collate_neural_input", &BinSim::token_seq_collate_neural_input);

    m.def("token_cfg_pack_neural_input", &BinSim::token_cfg_pack_neural_input);
    m.def("token_cfg_collate_neural_input", &BinSim::token_cfg_collate_neural_input);

    m.def("jtrans_seq_pack_neural_input", &BinSim::jtrans_seq_pack_neural_input);
    m.def("jtrans_seq_cfg_collate_neural_input", &BinSim::jtrans_seq_cfg_collate_neural_input);

    m.def("code_ast_pack_neural_input", &BinSim::code_ast_pack_neural_input);
    m.def("code_ast_collate_neural_input", &BinSim::code_ast_collate_neural_input);

    m.def("ins_dag_pack_neural_input", &BinSim::ins_dag_pack_neural_input);
    m.def("ins_dag_collate_neural_input", &BinSim::ins_dag_collate_neural_input);

    m.def("ins_cfg_pack_neural_input", &BinSim::ins_cfg_pack_neural_input);
    m.def("ins_cfg_collate_neural_input", &BinSim::ins_cfg_collate_neural_input);
    m.def("ins_cfg_preprocess_neural_input", &BinSim::ins_cfg_preprocess_neural_input);

    m.def("ins_seq_pack_neural_input", &BinSim::ins_seq_pack_neural_input,
          "Serialize neural input for instruction sequence.", py::arg("features"), py::arg("imm_values"));
    m.def("ins_seq_collate_neural_input", &BinSim::ins_seq_collate_neural_input,
          "Collate neural input for instruction sequence.", py::arg("inputs"), py::arg("chunks")=-1, py::arg("pack")=false);
}

int main(){
   int n, m;
    vector<tuple<uint16_t, uint16_t>> groups;
    std::cin >> n >> m;
    for(int i = 0; i < n; i++){
        uint16_t a, b;
        std::cin >> a >> b;
        groups.emplace_back(a, b);
    }
    return 0;
}