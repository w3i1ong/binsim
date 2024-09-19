#include "cfg_utils.h"
#include "../utils/InsManager.h"

py::bytes BinSim::ins_seq_pack_neural_input(const vector<uint16_t> &features, const vector<float> &imm_values) {
    SimpleSerializer serializer(4096);
    serializer.write((char)INS_SEQ);
    serializer.write(features);
    serializer.write(imm_values);
    return serializer.get_bytes();
}


tuple<map<string,py::array>,vector<py::array>>
BinSim::ins_seq_collate_neural_input(const vector<string>& inputs, int chunks, bool pack){
    vector<vector<int> > basic_blocks;
    vector<int> lengths;
    InstructionManager manager;
    for(auto & input : inputs){
        SimpleDeserializer serializer(input.data(), input.size());
        char data_type = serializer.read<char>();
        binsim_assert(data_type == INS_SEQ, "Invalid data type %d for ins_seq_collate_neural_input", data_type);

        vector<uint16_t> features = serializer.read<vector<uint16_t>>();
        vector<float> imm_values = serializer.read<vector<float>>();

        auto basic_block_ids = manager.add_basic_block(features, imm_values);
        basic_blocks.push_back(basic_block_ids);
        lengths.push_back((int)basic_block_ids.size());
    }
    if(pack){
        int total_length = 0;
        for(auto & length : lengths){
            total_length += length;
        }
        py::array_t<int> features_array(total_length);
        auto features_array_ptr = features_array.mutable_data();
        for(auto & basic_block : basic_blocks){
            memcpy(features_array_ptr, basic_block.data(), basic_block.size()*sizeof(int));
            features_array_ptr += basic_block.size();
        }
        return {
                manager.get_instruction_features(),
                vector<py::array>{features_array, vector_to_array(lengths)}
        };
    }

    if(chunks <= 1) {
        return {
                manager.get_instruction_features(),
                vector<py::array>{vector_to_array(basic_blocks), vector_to_array(lengths)}
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
    return {
            manager.get_instruction_features(),
            results
    };
}
