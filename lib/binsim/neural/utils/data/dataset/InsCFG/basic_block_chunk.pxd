cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        pass

cdef extern from "basic_block_chunk.cpp":
    pass


cdef extern from "basic_block_chunk.h" namespace "InsCFG":
    cdef vector[int] solve(vector[int]& nums, int k)
