#include "cfg_utils.h"
#include "../utils/InsManager.h"

py::bytes BinSim::ins_dag_pack_neural_input(uint16_t node_num, const vector<uint16_t> &src, const vector<uint16_t> &dst,
                                            const vector<uint16_t> &node_id,
                                            const vector<vector<uint16_t>> &features, const vector<float> &imm_values) {
    SimpleSerializer serializer(4096);
    serializer.write((char)INS_DAG);
    serializer.write(node_num);
    serializer.write(src);
    serializer.write(dst);
    serializer.write(node_id);
    serializer.write(features);
    serializer.write(imm_values);
    return serializer.get_bytes();
}


tuple<map<string,py::array>,vector<pair<py::array, py::array>>, vector<py::array>>
    BinSim::ins_dag_collate_neural_input(const vector<string>& inputs, int chunks){
    vector<pair<py::array, py::array>> edges;
    vector<int> node_nums, lengths, node_ids;
    vector<vector<int> > basic_blocks;
    vector<vector<uint16_t>> src_list, dst_list;
    InstructionManager manager;
    int base = 0;
    {
        py::gil_scoped_release release;
        for(auto & input : inputs){
            SimpleDeserializer serializer(input.data(), input.size());
            char data_type = serializer.read<char>();
            binsim_assert(data_type == INS_DAG, "Invalid data type %d for ins_dag_collate_neural_input", data_type);
            uint16_t node_num = serializer.read<uint16_t>();
            node_nums.push_back(node_num);
            src_list.push_back(serializer.read<vector<uint16_t>>());
            dst_list.push_back(serializer.read<vector<uint16_t>>());
            vector<uint16_t> saved_node_id = serializer.read<vector<uint16_t>>();
            // when batch size is large, the node id may be overflow during processing, so we need to use int instead of uint16_t.
            vector<int> node_id(saved_node_id.begin(), saved_node_id.end());
            binsim_assert(node_num == (int) node_id.size(), "Expect node_num(%d) == node_id.size(%zu)", node_num, node_id.size());

            int max_node_id = *max_element(node_id.begin(), node_id.end());
            for(auto & id : node_id){
                id += base;
            }
            base += (max_node_id + 1);
            node_ids.insert(node_ids.end(), node_id.begin(), node_id.end());

            vector<vector<uint16_t>> features = serializer.read<vector<vector<uint16_t>>>();
            binsim_assert((int)features.size() == max_node_id + 1, "Expect features.size(%zu) == max_node_id(%d) + 1", features.size(), max_node_id);
            vector<float> imm_values = serializer.read<vector<float>>();

            for(auto & basic_block : features){
                auto basic_block_ids = manager.add_basic_block(basic_block, imm_values);
                basic_blocks.push_back(basic_block_ids);
                lengths.push_back((int)basic_block_ids.size());
            }
        }
    }
    for(int i = 0;i < src_list.size(); i++){
        edges.emplace_back(vector_to_array(src_list[i]), vector_to_array(dst_list[i]));
    }
    if(chunks <= 1){
        return {
            manager.get_instruction_features(),
            edges,
            vector<py::array>{vector_to_array(basic_blocks), vector_to_array(lengths),
                              vector_to_array(node_nums), vector_to_array(node_ids)}
        };
    }

    vector<int> indexes;
    vector<vector<vector<int>>> result_chunks;
    vector<vector<int>> chunk_lengths;
    split_chunks(basic_blocks, chunks, result_chunks, chunk_lengths, indexes);
    vector<py::array> results;
    for(size_t i = 0; i < result_chunks.size(); i++){
        results.push_back(vector_to_array(result_chunks[i]));
        results.push_back(vector_to_array(chunk_lengths[i]));
    }
    results.push_back(vector_to_array(indexes));
    results.push_back(vector_to_array(node_nums));
    results.push_back(vector_to_array(node_ids));
    return {
        manager.get_instruction_features(),
        edges,
        results
    };
}
