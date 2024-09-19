#include "cfg_utils.h"
#include "../utils/InsManager.h"

py::bytes BinSim::ins_cfg_pack_neural_input(const vector<vector<uint16_t>> &adjList,
                                            const vector<vector<uint16_t>> &features, const vector<float> &imm_values) {
    SimpleSerializer serializer(4096);
    serializer.write((char)INS_CFG);
    serializer.write(adjList);
    serializer.write(features);
    serializer.write(imm_values);
    return serializer.get_bytes();
}

// compute the shortest path between all pairs of nodes
vector<vector<int16_t>> warshall(const vector<vector<uint16_t>> &adjList) {
    int node_num = (int)adjList.size();
    vector<vector<int16_t >> result(node_num, vector<int16_t>(node_num, -1));
    for (int i = 0; i < node_num; i++) {
        for (auto &j : adjList[i]) {
            result[i][j] = 1;
        }
        result[i][i] = 0;
    }
    for (int k = 0; k < node_num; k++) {
        for (int i = 0; i < node_num; i++) {
            for (int j = 0; j < node_num; j++) {
                if(result[i][k] != -1 && result[k][j] != -1 && (result[i][k] + result[k][j] < result[i][j] || result[i][j] == -1)){
                    result[i][j] = result[i][k] + result[k][j];
                }
            }
        }
    }
    return result;
}

vector<vector<int16_t >> warshall_with_super_source(const vector<vector<uint16_t >> &adjList){
    auto in_degree = vector<uint16_t>(adjList.size(), 0);
    for(auto & neighbors : adjList){
        for(auto neighbor : neighbors){
            in_degree[neighbor]++;
        }
    }
    auto new_adjList = vector<vector<uint16_t>>(adjList.size() + 1);
    for(int i = 0; i < (int)in_degree.size(); i++){
        if(in_degree[i] == 0) {
            new_adjList[0].push_back(i+1);
        }
    }
    for(int i = 0; i < (int)adjList.size(); i++){
        new_adjList[i+1] = adjList[i];
        for(auto & neighbor : new_adjList[i+1]){
            neighbor++;
        }
    }
    auto dis_matrix = warshall(new_adjList);
    int node_num = (int)adjList.size();
    for(int i = 0; i < node_num; i++){
        dis_matrix[i+1][0] = dis_matrix[0][i+1] = -2;
    }
    dis_matrix[0][0] = 0;
    return dis_matrix;
}

py::bytes BinSim::ins_cfg_preprocess_neural_input(const string& data){
    SimpleDeserializer deserializer(data.data(), data.size());
    char data_type = deserializer.read<char>();
    binsim_assert(data_type == INS_CFG, "Invalid data type %d for ins_cfg_preprocess_neural_input", data_type);
    auto adjList = deserializer.read<vector<vector<uint16_t>>>();
    auto features = deserializer.read<vector<vector<uint16_t>>>();
    auto imm_values = deserializer.read<vector<float>>();

    auto dist = warshall_with_super_source(adjList);
    SimpleSerializer serializer(data.size());
    serializer.write((char)INS_CFG);
    serializer.write(dist);
    serializer.write(features);
    serializer.write(imm_values);
    return serializer.get_bytes();
}

vector<py::array> collate_graphs(const vector<vector<vector<int16_t>>>& dist_matrix){
    int max_node_num = 0, total_node_num = 0;
    for(auto & graph : dist_matrix){
        max_node_num = max(max_node_num, (int)graph.size());
        // ignore the super source node, as it is not a real node
        total_node_num += (int)graph.size() - 1;
    }

    vector<int> new_node_idx(total_node_num, 0);
    int base = 0;
    for(int graph_idx = 0, node_idx=0; graph_idx < (int)dist_matrix.size(); graph_idx++){
        for(int k = 1; k < (int)dist_matrix[graph_idx].size(); k++){
            new_node_idx[node_idx++] = base + k;
        }
        base += max_node_num;
    }

    py::array_t<int> relative_distant({(int)dist_matrix.size(), max_node_num, max_node_num});
    auto ptr = relative_distant.mutable_data(0);
    memset(ptr, 0xff, (int)dist_matrix.size() * max_node_num * max_node_num * sizeof(int));
    for(const auto & graph_idx : dist_matrix){
        for(auto line: graph_idx){
            std::copy(line.begin(), line.end(), ptr);
            ptr += max_node_num;
        }
        for(size_t i = graph_idx.size(); i < max_node_num; i++) {
            ptr[i] = 0;
            ptr += max_node_num;
        }
    }
    return {vector_to_array(new_node_idx), relative_distant};
}


tuple<map<string,py::array>,vector<pair<py::array, py::array>>, vector<py::array>>
BinSim::ins_cfg_collate_neural_input(const vector<string>& inputs, int chunks){
    vector<int> lengths;
    vector<vector<int> > basic_blocks;
    vector<vector<vector<int16_t>>> dist_matrix;
    InstructionManager manager;
    for(auto & input : inputs){
        SimpleDeserializer serializer(input.data(), input.size());
        char data_type = serializer.read<char>();
        binsim_assert(data_type == INS_CFG, "Invalid data type %d for ins_dag_collate_neural_input", data_type);
        dist_matrix.push_back(serializer.read<vector<vector<int16_t >>>());
        // build instruction sequence for all basic blocks
        vector<vector<uint16_t>> features = serializer.read<vector<vector<uint16_t>>>();
        vector<float> imm_values = serializer.read<vector<float>>();
        for(auto & basic_block : features){
            auto basic_block_ids = manager.add_basic_block(basic_block, imm_values);
            basic_block_ids.insert(basic_block_ids.begin(), -1);
            basic_blocks.push_back(basic_block_ids);
            lengths.push_back((int)basic_block_ids.size());
        }
    }

    vector<vector<vector<int>>> result_chunks;
    vector<vector<int>> chunk_lengths;
    vector<int> indexes;
    split_chunks(basic_blocks, chunks, result_chunks, chunk_lengths, indexes);
    vector<pair<py::array, py::array>> results;
    for(size_t i = 0; i < result_chunks.size(); i++){
        auto packed_chunk = vector_to_array(result_chunks[i]);
        auto packed_length = vector_to_array(chunk_lengths[i]);
        results.emplace_back(packed_chunk, packed_length);
    }

    auto batched_graph = collate_graphs(dist_matrix);
    return {
        manager.get_instruction_features(),
        results,
        vector<py::array>({vector_to_array(indexes),
                           batched_graph[0],
                           batched_graph[1]
                          })
    };

}