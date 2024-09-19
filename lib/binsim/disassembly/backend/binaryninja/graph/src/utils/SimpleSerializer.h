#ifndef SIMPLE_SERIALIZER_H
#define SIMPLE_SERIALIZER_H
#include <vector>
#include <string>
#include <type_traits>
#include <cstring>
#include <pybind11/pybind11.h>
#include "./utils.h"
using namespace std;

template<class T>
using is_vector = is_same<T, vector<typename T::value_type>>;

namespace py = pybind11;
class SimpleSerializer {
private:
    char* buffer;
    size_t total_size;
    size_t capacity;
    void ensure_at_least(size_t size);
public:
    explicit SimpleSerializer(size_t size);
    ~SimpleSerializer();
    py::bytes get_bytes();

    size_t write(const string& value){
        return write(value.data(),(int) value.size());
    }

    template<class T>
    typename enable_if<is_arithmetic<T>::value, size_t>::type
    write(T value){
        ensure_at_least(sizeof(T));
        memcpy(buffer + total_size, &value, sizeof(T));
        total_size += sizeof(T);
        return sizeof(T);
    }

    template<class T>
    typename enable_if<is_arithmetic<T>::value, size_t>::type
    write(const vector<T>& vec){
        return write(vec.data(), (int) vec.size());
    }

    template<class T>
    typename enable_if<is_arithmetic<T>::value, size_t>::type
    write(const vector<vector<T>>& matrix){
        size_t bytes_num = write((int) matrix.size());
        for(auto& vec: matrix){
            bytes_num += write(vec);
        }
        return bytes_num;
    }

    template<class T>
    typename enable_if<is_arithmetic<T>::value, size_t>::type
    write(T* ptr, int size){
        size_t bytes_num = sizeof(int) + size * sizeof(T);
        ensure_at_least(bytes_num);
        // Write the size of the vector
        memcpy(buffer + total_size, &size, sizeof(int));
        total_size += sizeof(int);
        // Write the vector
        memcpy(buffer + total_size, ptr, size * sizeof(T));
        total_size += size * sizeof(T);

        return bytes_num;
    }
};


class SimpleDeserializer {
private:
    const char* buffer;
    size_t total_size;
    size_t cur_ptr;


    void check_remaining_bytes(size_t size) const{
        binsim_assert(cur_ptr + size <= total_size, "Expect %ld bytes, but only %ld bytes left", size, total_size - cur_ptr);
    }

public:
    explicit SimpleDeserializer(const char* buffer, size_t size): buffer(buffer), total_size(size), cur_ptr(0){}

    template<class T>
    typename enable_if<is_arithmetic<T>::value, T>::type read(){
        T result;
        check_remaining_bytes(sizeof(T));
        memcpy(&result, buffer + cur_ptr, sizeof(T));
        cur_ptr += sizeof(T);
        return result;
    }

    template<class T>
    typename enable_if<is_same<T, string>::value, T>::type
    read(){
        int size = read<int>();
        check_remaining_bytes(size);
        string result(buffer + cur_ptr, size);
        cur_ptr += size;
        return result;
    }


    template<class T>
    typename enable_if<is_vector<T>::value && is_arithmetic<typename T::value_type>::value, T>::type
    read(){
        int size = read<int>();
        T result(size);
        check_remaining_bytes(size * sizeof(typename T::value_type));
        memcpy(result.data(), buffer + cur_ptr, size * sizeof(typename T::value_type));
        cur_ptr += size * sizeof(typename T::value_type);
        return result;
    }

    template<class T>
    typename enable_if<is_vector<T>::value && is_vector<typename T::value_type>::value &&
        is_arithmetic<typename T::value_type::value_type>::value, T>::type
    read(){
        int size = read<int>();
        T result(size);
        for(auto& vec: result){
            vec = read<typename T::value_type>();
        }
        return result;
    }

    int getSize();
    int getRemainSize();
};

#endif //SIMPLE_SERIALIZER_H
