#include "cfg_utils.h"

py::bytes BinSim::bytecode_pack_neural_input(uint in_degree, uint out_degree, const string& payload) {
    SimpleSerializer serializer(1024);
    serializer.write((char)BYTECODE);
    serializer.write(in_degree);
    serializer.write(out_degree);
    serializer.write(payload);
    return serializer.get_bytes();
}

tuple<py::array, py::array> BinSim::bytecode_collate_neural_input(const vector<string >& inputs, int max_byte_num) {
    py::array_t<uint> degrees({(int)inputs.size(), 2});
    py::array_t<int8_t> payload({(int)inputs.size(), max_byte_num});
    {
        py::gil_scoped_release release;
        auto degrees_ptr = degrees.mutable_data(0);
        auto payload_ptr = payload.mutable_data(0);
        memset(payload_ptr, 0, inputs.size() * max_byte_num);
        for(auto& input: inputs){
            SimpleDeserializer deserializer(input.data(), input.size());
            char data_type = deserializer.read<char>();
            binsim_assert(data_type == BYTECODE, "Expect the magic number of BYTECODE(%d), but got %d", BYTECODE, data_type);
            uint in_degree = deserializer.read<uint>();
            uint out_degree = deserializer.read<uint>();
            *degrees_ptr++ = in_degree;
            *degrees_ptr++ = out_degree;

            vector<int8_t> payload_vec = deserializer.read<vector<int8_t >>();
            int copied_size = min(max_byte_num, (int)payload_vec.size());
            memcpy(payload_ptr, payload_vec.data(), copied_size);
            payload_ptr += max_byte_num;
        }
    }
    return std::make_tuple(degrees, payload);
}
