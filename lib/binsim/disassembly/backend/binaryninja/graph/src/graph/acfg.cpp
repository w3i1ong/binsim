#include "cfg_utils.h"

py::bytes BinSim::acfg_pack_neural_input(uint16_t node_num,
                                         const vector<uint16_t>& src,
                                         const vector<uint16_t>& dst,
                                         const vector<vector<float>>&features){
    int edge_num = (int) src.size();
    int fea_dim = (int) features[0].size();
    SimpleSerializer serializer(1024);
    serializer.write((char)ACFG);
    serializer.write(node_num);
    serializer.write(edge_num);
    serializer.write(fea_dim);
    serializer.write(src);
    serializer.write(dst);
    serializer.write(features);
    return serializer.get_bytes();
}

tuple<vector<tuple<py::array, py::array>>, py::array, py::array>
        BinSim::acfg_collate_neural_input(const vector<string>& inputs){
    vector<tuple<py::array, py::array>> edges;
    vector<vector<float>> features;
    vector<vector<uint16_t>> src_list, dst_list;
    vector<uint16_t> node_nums;
    {
        py::gil_scoped_release release;
        for(auto& input: inputs){
            SimpleDeserializer deserializer(input.data(), input.size());
            char data_type = deserializer.read<char>();
            binsim_assert(data_type == ACFG, "Expect the magic number of ACFG(%d), but got %d", ACFG, data_type);
            uint16_t node_num = deserializer.read<uint16_t>();
            deserializer.read<int>();
            deserializer.read<int>();
            node_nums.push_back(node_num);

            src_list.push_back(deserializer.read<vector<uint16_t>>());
            dst_list.push_back(deserializer.read<vector<uint16_t>>());

            vector<vector<float>> fea = deserializer.read<vector<vector<float>>>();
            for(auto& f: fea){
                features.push_back(f);
            }
        }
    }
    for(int i=0;i<src_list.size() ;i ++){
        py::array_t<uint16_t> src_array((int)src_list[i].size(), src_list[i].data());
        py::array_t<uint16_t> dst_array((int)dst_list[i].size(), dst_list[i].data());
        edges.emplace_back(src_array, dst_array);
    }
    py::array_t<float> features_array = vector_to_array(features);
    py::array_t<uint16_t> node_nums_array((int)node_nums.size(), node_nums.data());
    return {edges, features_array, node_nums_array};
}
