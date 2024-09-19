#include "cfg_utils.h"


py::bytes BinSim::token_seq_pack_neural_input(const vector<int>&features, int bb_max_len){
    int copied_size = min(bb_max_len, (int)features.size());
    SimpleSerializer serializer(1024);
    serializer.write((char)TOKEN_SEQ);
    serializer.write(features.data(), copied_size);
    return serializer.get_bytes();
}


vector<py::array> merge_token_seq(const vector<string>& inputs){
    int max_bb_len = 0;
    vector<vector<int>> features;
    vector<int> basic_block_lengths;
    {
        py::gil_scoped_release release;
        features.reserve(inputs.size());
        for(auto& input: inputs){
            SimpleDeserializer deserializer(input.data(), input.size());
            char data_type = deserializer.read<char>();
            binsim_assert(data_type == TOKEN_SEQ, "Expect the magic number of TOKEN_SEQ(%d), but got %d", TOKEN_SEQ, data_type);
            vector<int> feature = deserializer.read<vector<int>>();
            features.push_back(feature);
            max_bb_len = max(max_bb_len, (int)feature.size());
            basic_block_lengths.push_back((int)feature.size());
        }
    }
    return {vector_to_array(features), vector_to_array(basic_block_lengths)};
}

vector<py::array> BinSim::token_seq_collate_neural_input(const vector<string>& inputs, int chunks, int max_length){
    if(chunks <= 1){
        return merge_token_seq(inputs);
    }
    // deserialize the input
    vector<vector<int> > features;
    for(auto& input: inputs){
        SimpleDeserializer deserializer(input.data(), input.size());
        char data_type = deserializer.read<char>();
        binsim_assert(data_type == TOKEN_SEQ, "Expect the magic number of TOKEN_SEQ(%d), but got %d", TOKEN_SEQ, data_type);
        vector<int> feature = deserializer.read<vector<int>>();
        feature.resize(max_length);
        features.push_back(feature);
    }
    // sort the input by the length of the features
    vector<int> index;
    vector<vector<int>> chunk_lengths;
    vector<vector<vector<int>>> result_chunks;
    split_chunks(features, chunks, result_chunks, chunk_lengths, index);
    vector<py::array> results;

    results.reserve(result_chunks.size()*2+1);
    for(size_t i = 0; i < result_chunks.size(); i++){
        results.push_back(vector_to_array(result_chunks[i]));
        results.push_back(vector_to_array(chunk_lengths[i]));
    }
    results.push_back(vector_to_array(index));
    return results;
}
