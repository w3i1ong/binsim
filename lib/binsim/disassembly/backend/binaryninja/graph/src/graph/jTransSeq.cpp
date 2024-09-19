#include "cfg_utils.h"

py::bytes BinSim::jtrans_seq_pack_neural_input(const vector<int>&features){
    SimpleSerializer serializer(4096);
    serializer.write((char)JTRANS_SEQ);
    serializer.write(features);
    return serializer.get_bytes();

}
vector<py::array> BinSim::jtrans_seq_cfg_collate_neural_input(const vector<string>& inputs){
    vector<vector<int>> features;
    vector<int> lengths;
    for(auto& input: inputs){
        SimpleDeserializer deserializer(input.data(), input.size());
        char data_type = deserializer.read<char>();
        binsim_assert(data_type == JTRANS_SEQ, "Expect the magic number of JTRANS_SEQ(%d), but got %d", JTRANS_SEQ, data_type);
        vector<int> feature = deserializer.read<vector<int>>();
        features.push_back(feature);
        lengths.push_back(feature.size());
    }
    return {vector_to_array(features), vector_to_array(lengths)};
}
