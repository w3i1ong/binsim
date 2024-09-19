cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        pass

cdef extern from "<string>" namespace "std":
    cdef cppclass string:
        pass

cdef extern from "<regex>" namespace "std":
    cdef cppclass regex:
        pass

cdef extern from "<set>" namespace "std":
    cdef cppclass set[T]:
        pass

cdef extern from "<map>" namespace "std":
    cdef cppclass map[T, U]:
        pass

cdef extern from "<utility>" namespace "std":
    cdef cppclass pair[T, U]:
        pass

cdef extern from "IDAIns.h":
    cdef cppclass IDAIns:
        IDAIns() nogil

        IDAIns(const map[string, int]& token2idx) nogil
        string getOperator() nogil
        vector[string] getOperands() nogil
        void parseIns(string ins) nogil

        void setToken2idx(const map[string, int]& token2idx) nogil

        map[string, int] getToken2idx() nogil

        vector[int] parseFunction(vector[string]& ins_list,
                                  size_t max_length, const vector[pair[int, int]]& addr2idx) nogil
