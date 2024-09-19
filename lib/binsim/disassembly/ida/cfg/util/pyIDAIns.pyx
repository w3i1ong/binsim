# distutils: language=c++
# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf-8
from .IDAIns cimport IDAIns
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from threading import Thread
from queue import Queue
import os
cdef class pyIDAIns:
    cdef IDAIns *objPtr

    def __cinit__(self):
        self.objPtr = new IDAIns()

    def __init__(self):
        # read tokenizer from file
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_dir}/data/tokenizer/vocab.txt") as f:
            vocab = f.read().strip().split("\n") + ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        vocab = dict((v, i) for i, v in enumerate(vocab))
        self.objPtr.setToken2idx(vocab)

    @property
    def vocab(self):
        return self.objPtr.getToken2idx()

    def __dealloc__(self):
        del self.objPtr

    cpdef getOperator(self) :
        return self.objPtr.getOperator()

    cpdef getOperands(self):
        return self.objPtr.getOperands()

    cpdef parseIns(self, string ins):
        self.objPtr.parseIns(ins)
        return self.objPtr.getOperator(), self.objPtr.getOperands()

    cdef vector[int] _parseFunction(self, vector[string] ins_list, size_t max_length, vector[pair[int, int]] bb_length) nogil:
        return self.objPtr.parseFunction(ins_list, max_length, bb_length)

    def parseFunction(self, list[str] ins_list, int max_length, vector[pair[int, int]] bb_length):
        return self._parseFunction(ins_list, max_length, bb_length)

    def parseFunctions(self, insListIterator, int max_length, num_threads, batch_size):
        if num_threads == 0:
            result = []
            for ins_list, bb_length in insListIterator:
                result.append(self._parseFunction(ins_list, max_length, bb_length))
            return result

        result = []
        input_queue = Queue()
        output_queue = Queue()
        threads = []
        threads.append(Thread(target=self.parseFunctions_producer, args=(input_queue, insListIterator, batch_size, num_threads)))
        for i in range(num_threads):
            threads.append(Thread(target=self.parseFunctions_consumer, args=(input_queue, max_length, output_queue)))
        threads.append(Thread(target=self.parseFunctions_merger, args=(output_queue, result, num_threads, batch_size)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return result

    cdef list[list[int]] _parseFunctions(self, list functions, int max_length):
        cdef vector[vector[string]] function_tokens
        cdef vector[vector[pair[int, int]]] bb_length
        cdef vector[vector[int]] result
        cdef int i

        function_tokens.reserve(len(functions))
        bb_length.reserve(len(functions))
        for ins_list, bb_len in functions:
            function_tokens.push_back(ins_list)
            bb_length.push_back(bb_len)

        with nogil:
            for i in range(function_tokens.size()):
                result.push_back(self._parseFunction(function_tokens[i], max_length, bb_length[i]))

        return result


    @staticmethod
    def parseFunctions_producer(input_queue, insListIterator, batch_size, num_threads):
        batched_insList = []
        idx = 0
        for insList in insListIterator:
            batched_insList.append(insList)
            if len(batched_insList) == batch_size:
                input_queue.put((idx, batched_insList))
                idx += 1
                batched_insList = []
        if len(batched_insList) > 0:
            input_queue.put((idx, batched_insList))

        for i in range(num_threads):
            input_queue.put(None)


    @staticmethod
    def parseFunctions_consumer(input_queue, max_length, output_queue):
        cdef pyIDAIns parser = pyIDAIns()
        cdef int idx
        while True:
            data = input_queue.get()
            if data is None:
                break
            idx, func_data = data
            output_queue.put((idx, parser._parseFunctions(func_data, max_length)))
        output_queue.put(None)

    @staticmethod
    def parseFunctions_merger(output_queue, list res, num_threads, batch_size):
        while True:
            data = output_queue.get()
            if data is None:
                num_threads -= 1
                if num_threads == 0:
                    break
                continue
            idx, result_data = data
            excepted_length = idx * batch_size + len(result_data)
            if excepted_length >= len(res):
                res.extend([None] * (excepted_length - len(res)))
            res[idx*batch_size:excepted_length] = result_data

