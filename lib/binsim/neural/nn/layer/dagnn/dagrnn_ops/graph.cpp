#include "graph.h"
#include <algorithm>
#include <chrono>
#include <pybind11/pybind11.h>
#define GRAPH_ASSERT(expression, message) do { \
        TORCH_CHECK((expression), #expression " check failed! at ", __FILE__, ":", __LINE__, " ", message) \
    }                                          \
    while(0)

void topology_batch(const vector<vector<int>>& ajd_list, vector<int> in_degrees,
                    vector<vector<int>>& node_batch_forward,
                    vector<vector<GraphEdge>>& edge_batch_forward,  vector<vector<int>>& edge_batch_index_forward,
                    vector<vector<GraphEdge>>& edge_batch_backward, vector<vector<int>>& edge_batch_index_backward){
    int total_node_num = (int)ajd_list.size();
    vector<int> node_batch, edge_batch_index;
    vector<GraphEdge> edge_batch;
    // first batch
    queue<int> q;
    for(int i=0; i<total_node_num; i++){
        if(in_degrees[i] == 0){
            q.push(i);
        }
    }
    // bfs
    while(!q.empty()){
        node_batch.clear();
        edge_batch.clear();
        int queue_length = (int)q.size();
        for(int i=0; i<queue_length; i++){
            int node = q.front();
            node_batch.push_back(node);
            q.pop();
            for(auto next_node: ajd_list[node]){
                in_degrees[next_node]--;
                if(in_degrees[next_node] == 0){
                    q.push(next_node);
                }
            }
        }
        node_batch_forward.push_back(node_batch);
        for(auto node: node_batch){
            for(auto next_node: ajd_list[node]){
                edge_batch.emplace_back(node, next_node);
            }
        }

        if(!edge_batch.empty()){
            edge_batch_index.clear();
            edge_batch_backward.push_back(edge_batch);
            edge_batch_index.clear();
            for(int i=0; i<(int)edge_batch.size(); i++){
                if( i == 0 || edge_batch[i].src != edge_batch[i-1].src){
                    edge_batch_index.push_back(i);
                }
            }
            edge_batch_index.push_back((int)edge_batch.size());
            edge_batch_index_backward.push_back(edge_batch_index);

            sort(edge_batch.begin(), edge_batch.end(), [](const GraphEdge& a, const GraphEdge& b){
                return a.dst < b.dst;
            });
            edge_batch_index.clear();
            edge_batch_forward.push_back(edge_batch);
            for(int i=0; i<(int)edge_batch.size(); i++){
                if( i == 0 || edge_batch[i].dst != edge_batch[i-1].dst){
                    edge_batch_index.push_back(i);
                }
            }
            edge_batch_index.push_back((int)edge_batch.size());
            edge_batch_index_forward.push_back(edge_batch_index);
        }
    }
}

tuple<vector<Tensor>, vector<Tensor>, vector<Tensor>, vector<Tensor>, vector<Tensor>, Tensor, Tensor>
        prepare_update_information_for_faster_forward(const vector<vector<int>>& adj_list){
    // release GIL
    py::gil_scoped_release release;
    int total_node_num = (int)adj_list.size();
    vector<int> in_degrees(total_node_num, 0), out_degrees(total_node_num, 0);
    for(int i=0; i<total_node_num; i++){
        out_degrees[i] = (int)adj_list[i].size();
        for(auto j: adj_list[i]){
            in_degrees[j]++;
            GRAPH_ASSERT(j < total_node_num, "Node index out of range.");
        }
    }
    vector<vector<int>> node_batch_forward;
    vector<vector<GraphEdge>> edge_batch_forward, edge_batch_backward;
    vector<vector<int>> edge_batch_index_forward, edge_batch_index_backward;
    topology_batch(adj_list, in_degrees,
                   node_batch_forward,
                   edge_batch_forward, edge_batch_index_forward,
                   edge_batch_backward, edge_batch_index_backward);
    vector<Tensor> node_batch_forward_tensor, edge_batch_forward_tensor, edge_batch_backward_tensor;
    vector<Tensor> edge_batch_index_forward_tensor, edge_batch_index_backward_tensor;

    int node_num_in_batch = 0;
    for(auto node_batch: node_batch_forward){
        node_batch_forward_tensor.push_back(torch::from_blob(node_batch.data(), {(int)node_batch.size()}, torch::kInt32).clone());
        node_num_in_batch += (int)node_batch.size();
    }
    GRAPH_ASSERT(node_num_in_batch == total_node_num, "Loop detected in the given graph.");
    for(auto edge_batch: edge_batch_forward){
        edge_batch_forward_tensor.push_back(torch::from_blob(edge_batch.data(), {(int)edge_batch.size(), 2}, torch::kInt32).clone());
    }
    for(auto edge_batch: edge_batch_backward){
        Tensor edge_batch_tensor = torch::from_blob(edge_batch.data(), {(int)edge_batch.size(), 2}, torch::kInt32).clone();
        edge_batch_tensor = torch::flip(edge_batch_tensor, {1});
        edge_batch_backward_tensor.push_back(edge_batch_tensor);
    }
    for(auto edge_batch_index: edge_batch_index_forward){
        edge_batch_index_forward_tensor.push_back(torch::from_blob(edge_batch_index.data(), {(int)edge_batch_index.size()}, torch::kInt32).clone());
    }
    for(auto edge_batch_index: edge_batch_index_backward){
        edge_batch_index_backward_tensor.push_back(torch::from_blob(edge_batch_index.data(), {(int)edge_batch_index.size()}, torch::kInt32).clone());
    }
    Tensor in_degrees_tensor = torch::from_blob(in_degrees.data(), {(int)in_degrees.size()}, torch::kInt32).clone();
    Tensor out_degrees_tensor = torch::from_blob(out_degrees.data(), {(int)out_degrees.size()}, torch::kInt32).clone();
    return std::make_tuple(node_batch_forward_tensor, edge_batch_forward_tensor, edge_batch_index_forward_tensor,
                           edge_batch_backward_tensor, edge_batch_index_backward_tensor,
                           in_degrees_tensor, out_degrees_tensor);
}

tuple<vector<vector<int>>, Tensor> transform_adj_list_for_fast_forward(const vector<vector<int>>& adj_list){
    // release GIL
    py::gil_scoped_release release;
    vector<vector<int>> new_adj_list(adj_list.size());
    vector<int> new_index(adj_list.size());
    vector<int> in_degrees(adj_list.size(), 0);
    queue<int> q;

    // Count the in-degree of each node.
    for(const auto & neighborhood : adj_list){
        for(auto & next: neighborhood){
            in_degrees[next]++;
        }
    }
    int cur_index = 0;
    for(int i=0; i<(int)adj_list.size(); i++){
        if(in_degrees[i] == 0){
            q.push(i);
        }
    }
    while(!q.empty()){
        int node = q.front();
        q.pop();
        new_index[node] = cur_index++;
        for(auto next_node: adj_list[node]){
            in_degrees[next_node]--;
            if(in_degrees[next_node] == 0){
                q.push(next_node);
            }
        }
    }
    GRAPH_ASSERT(cur_index == (int)adj_list.size(), "Loop detected in the given graph.");
    for(int i=0; i<(int)adj_list.size(); i++){
        GRAPH_ASSERT(new_adj_list[new_index[i]].empty(),
                     "Unexpected error: Expecting empty list, bue get a non-empty list with size "
                     + std::to_string(new_adj_list[new_index[i]].size()));
        new_adj_list[new_index[i]].reserve(adj_list[i].size());
        for(auto j: adj_list[i]){
            new_adj_list[new_index[i]].push_back(new_index[j]);
        }
    }
    return make_tuple(new_adj_list, torch::from_blob(new_index.data(), {(int)new_index.size()}, torch::kInt32).clone());
}


std::vector<std::vector<int>> create_adj_list(const std::vector<std::vector<py::array>>& edges, py::array_t<int> node_nums) {
    py::gil_scoped_release release;  // Release GIL

    // 获取节点总数
    auto node_nums_unchecked = node_nums.unchecked<1>();
    int total_nodes = 0;
    for (ssize_t i = 0; i < node_nums_unchecked.size(); ++i) {
        total_nodes += node_nums_unchecked(i);
    }

    // 初始化邻接表
    std::vector<std::vector<int>> adj_list(total_nodes);

    int base = 0;
    for (size_t i = 0; i < edges.size(); ++i) {
        auto& edge_pair = edges[i];
        auto src_list = edge_pair[0].cast<py::array_t<int>>();
        auto dst_list = edge_pair[1].cast<py::array_t<int>>();

        auto src_unchecked = src_list.unchecked<1>();
        auto dst_unchecked = dst_list.unchecked<1>();

        for (ssize_t j = 0; j < src_unchecked.size(); ++j) {
            adj_list[src_unchecked(j) + base].emplace_back(dst_unchecked(j) + base);
        }

        base += node_nums_unchecked(i);
    }

    return adj_list;
}
