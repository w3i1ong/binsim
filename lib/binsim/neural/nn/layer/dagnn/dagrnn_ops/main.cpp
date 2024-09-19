#include <torch/extension.h>
#include "daggru.h"
#include "graph.h"

PYBIND11_MODULE(dagrnn_ops, m){
    m.def("message_passing_forward", &message_passing_forward);
    m.def("fused_gru_partial_forward", &fused_gru_partial_forward);
    m.def("fused_gru_partial_backward", &fused_gru_partial_backward);
    m.def("fused_lstm_partial_forward", &fused_lstm_partial_forward,
        pybind11::arg("last_cell"), pybind11::arg("last_hidden"),
        pybind11::arg("cell"), pybind11::arg("hidden"),
        pybind11::arg("i_gate"), pybind11::arg("f_gate"), pybind11::arg("g_gate"), pybind11::arg("o_gate"));
    m.def("fused_lstm_partial_backward", &fused_lstm_partial_backward,
        pybind11::arg("grad_cell"), pybind11::arg("grad_hidden"),
        pybind11::arg("grad_last_cell"), pybind11::arg("grad_last_hidden"),
        pybind11::arg("last_cell"), pybind11::arg("last_hidden"),
        pybind11::arg("i"), pybind11::arg("f"), pybind11::arg("g"), pybind11::arg("o"));
    m.def("prepare_update_information_for_faster_forward", &prepare_update_information_for_faster_forward,
          pybind11::arg("adj_list"));
    m.def("transform_adj_list_for_fast_forward", &transform_adj_list_for_fast_forward,
          pybind11::arg("adj_list"));
    m.def("create_adj_list", &create_adj_list, "Create adjacency list from edges and node numbers");
}
