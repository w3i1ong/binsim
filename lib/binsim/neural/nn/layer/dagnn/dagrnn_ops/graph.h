#ifndef GRAPH_H
#define GRAPH_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <tuple>
#include <torch/torch.h>
#include <torch/extension.h>
using std::vector;
using std::queue;
using std::tuple;
using torch::Tensor;

struct GraphEdge{
    int src;
    int dst;
    GraphEdge(int src, int dst): src(src), dst(dst){}
};

#define GRAPH_ASSERT(expression, message) do { \
        TORCH_CHECK((expression), #expression " check failed! at ", __FILE__, ":", __LINE__, " ", message) \
    }                                          \
    while(0)
void topology_batch(const vector<vector<int>>& ajd_list, vector<int> in_degrees,
                    vector<vector<int>>& node_batch_forward,
                    vector<vector<GraphEdge>>& edge_batch_forward,  vector<vector<int>>& edge_batch_index_forward,
                    vector<vector<GraphEdge>>& edge_batch_backward, vector<vector<int>>& edge_batch_index_backward);

tuple<vector<Tensor>, vector<Tensor>, vector<Tensor>, vector<Tensor>, vector<Tensor>, Tensor, Tensor>
        prepare_update_information_for_faster_forward(const vector<vector<int>>& adj_list);

tuple<vector<vector<int>>, Tensor> transform_adj_list_for_fast_forward(const vector<vector<int>>& adj_list);
std::vector<std::vector<int>> create_adj_list(const std::vector<std::vector<py::array>>& edges, py::array_t<int> node_nums);
#endif
