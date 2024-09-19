#include "cfg_utils.h"
#include "iostream"
py::bytes BinSim::token_cfg_pack_neural_input(uint16_t node_num,
                                      const vector<uint16_t>& src,
                                      const vector<uint16_t>& dst,
                                      const vector<vector<int>>&features){
    SimpleSerializer serializer(4096);
    serializer.write((char)TOKEN_CFG);
    serializer.write(node_num);
    serializer.write(src);
    serializer.write(dst);
    serializer.write(features);
    return serializer.get_bytes();
}


pair<vector<pair<py::array, py::array>>, vector<py::array>>
        BinSim::token_cfg_collate_neural_input(const vector<std::string> &inputs, int chunks) {
    // deserialize the input
    vector<pair<py::array, py::array>> edges;
    vector<vector<int>> features;
    vector<uint16_t> node_nums;
    for(auto& input: inputs){
        SimpleDeserializer deserializer(input.data(), input.size());
        char data_type = deserializer.read<char>();
        binsim_assert(data_type == TOKEN_CFG, "Expect the magic number of TOKEN_CFG(%d), but got %d", TOKEN_CFG, data_type);
        uint16_t node_num = deserializer.read<uint16_t>();
        node_nums.push_back(node_num);

        vector<uint16_t> src = deserializer.read<vector<uint16_t>>();
        vector<uint16_t> dst = deserializer.read<vector<uint16_t>>();
        edges.emplace_back(vector_to_array(src), vector_to_array(dst));

        vector<vector<int>> feature = deserializer.read<vector<vector<int>>>();
        features.insert(features.end(), feature.begin(), feature.end());
    }
    if(chunks <= 1){
        vector<int> lengths;
        lengths.reserve(features.size());
        for(auto& feature: features){
            lengths.push_back((int)feature.size());
        }
        return {edges, vector<py::array>({vector_to_array(features), vector_to_array(lengths), vector_to_array(node_nums)})};
    }

    vector<int> index;
    vector<vector<int>> lengths;
    vector<vector<vector<int>>> result_chunks;
    split_chunks(features, chunks, result_chunks, lengths, index);
    vector<py::array> results;
    results.reserve(result_chunks.size()*2 + 1);
    for(int i = 0; i < result_chunks.size(); i++){
        results.push_back(vector_to_array(result_chunks[i]));
        results.push_back(vector_to_array(lengths[i]));
    }
    results.push_back(vector_to_array(index));
    return {edges, results};
}
