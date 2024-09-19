#include "SimpleSerializer.h"
#include "./utils.h"
void SimpleSerializer::ensure_at_least(size_t size) {
    if(total_size + size < capacity){
        return;
    }
    size_t new_capacity = std::max(2 * capacity, capacity + size);
    char* new_buffer = new char[new_capacity];
    binsim_assert(new_buffer != nullptr, "Failed to allocate memory for SimpleSerializer");
    memcpy(new_buffer, buffer, total_size);
    delete[] buffer;
    buffer = new_buffer;
    capacity = new_capacity;
}

SimpleSerializer::~SimpleSerializer() {
    delete[] buffer;
}

SimpleSerializer::SimpleSerializer(size_t size) {
    buffer = new char[size];
    total_size = 0;
    capacity = size;
}

py::bytes SimpleSerializer::get_bytes() {
    py::bytes result(buffer, total_size);
    return result;
}



int SimpleDeserializer::getSize(){
    return this->total_size;
}

int SimpleDeserializer::getRemainSize(){
    return this->total_size - this->cur_ptr;
}
