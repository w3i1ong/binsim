#include "cfg_utils.h"

py::bytes BinSim::code_ast_pack_neural_input(const vector<int>& src, const vector<int>& dst,
                                             int node_num, int callee_num,
                                             const vector<int>&features){
    SimpleSerializer serializer(4096);
    serializer.write((char)CODE_AST);
    serializer.write(node_num);
    serializer.write(callee_num);
    serializer.write(src);
    serializer.write(dst);
    serializer.write(features);
    return serializer.get_bytes();
}

tuple<vector<pair<py::array, py::array>>, vector<py::array>> BinSim::code_ast_collate_neural_input(
        const vector<string>& inputs){
    vector<pair<py::array, py::array>> edges;
    vector<int> node_nums, features, callee_nums;
    vector<vector<int>> src_list, dst_list;
    {
        py::gil_scoped_release release;
        for(auto& input: inputs){
            SimpleDeserializer deserializer(input.data(), input.size());
            char data_type = deserializer.read<char>();
            binsim_assert(data_type == CODE_AST, "Expect the magic number of CODE_AST(%d), but got %d", CODE_AST, data_type);
            int node_num = deserializer.read<int>();
            node_nums.push_back(node_num);
            int callee_num = deserializer.read<int>();
            callee_nums.push_back(callee_num);

            src_list.push_back(deserializer.read<vector<int>>());
            dst_list.push_back(deserializer.read<vector<int>>());
            vector<int> feature = deserializer.read<vector<int>>();

            binsim_assert("%d\n", "Meet error while deserialize code AST.");
            features.insert(features.end(), feature.begin(), feature.end());
        }
    }

    for(int i = 0; i < src_list.size(); i++){
        edges.emplace_back(vector_to_array(src_list[i]), vector_to_array(dst_list[i]));
    }

    return {edges, vector<py::array>({vector_to_array(features),
                                      vector_to_array(node_nums),
                                      vector_to_array(callee_nums)})};
}
