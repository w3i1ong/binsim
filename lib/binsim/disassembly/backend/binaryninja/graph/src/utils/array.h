#ifndef DATAUTILS_ARRAY_H
#define DATAUTILS_ARRAY_H
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
namespace py = pybind11;
using namespace std;

template<class T>
typename enable_if<is_arithmetic<T>::value, py::array_t<T>>::type
vector_to_array(const vector<T>& vec) {
    py::array_t<T> result((int)vec.size());
    auto ptr = result.mutable_data(0);
    memcpy(ptr, vec.data(), vec.size() * sizeof(T));
    return result;
}

template <typename T>
typename enable_if<is_arithmetic<T>::value, py::array_t<T>>::type
vector_to_array(const std::vector<std::vector<T>>& vec, int max_column=-1, int start_row=-1, int end_row=-1) {
    binsim_assert((start_row >= 0 && end_row >= 0 && start_row < end_row && end_row <= (int)vec.size())|| (start_row == -1 && end_row == -1),
                  "Invalid start_row(%d) or end_row(%d) for vector_to_array", start_row, end_row);
    if(start_row == -1){
        start_row = 0;
        end_row = vec.size();
    }

    int real_max_column = 0;
    for(int i = start_row; i < end_row; i++){
        auto & row = vec[i];
        real_max_column = max(real_max_column, (int)row.size());
    }
    if(max_column > real_max_column || max_column < 0){
        max_column = real_max_column;
    }

    py::array_t<T> result({end_row-start_row, max_column});
    auto ptr = result.mutable_data(0);
    memset(ptr, 0, (end_row-start_row) * max_column * sizeof(T));
    for(int i = start_row; i < end_row; i++){
        auto& line = vec[i];
        int copy_num = min((int)line.size(), max_column);
        memcpy(ptr, line.data(), copy_num * sizeof(T));
        ptr += max_column;
    }
    return result;
}

template<class T>
typename enable_if<is_arithmetic<T>::value, py::array_t<T>>::type
flatten_vector_to_array(const vector<vector<T>>& vec){
    size_t total_length = 0;
    for(auto& line: vec){
        total_length += line.size();
    }
    py::array_t<T> result(total_length);
    auto ptr = result.mutable_data(0);
    for(auto& line: vec){
        memcpy(ptr, line.data(), line.size() * sizeof(T));
        ptr += line.size();
    }
    return result;
}


#endif //DATAUTILS_ARRAY_H
