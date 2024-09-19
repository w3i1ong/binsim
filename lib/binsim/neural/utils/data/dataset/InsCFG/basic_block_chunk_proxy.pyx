# distutils: language=c++
# from basic_block_chunk cimport solve
from libcpp.vector cimport vector

cdef struct DPState:
    int padding
    int last_chunk_end


cpdef py_solve(list lengths, int max_chunk_num, max_padding_ratio=None):
    cdef int chunk_num, i, j, l, cum_padding, padding, max_padding
    cdef vector[vector[DPState]] dp

    basic_block_num = len(lengths)
    dp.resize(basic_block_num)
    for i in range(basic_block_num):
        dp[i].resize(max_chunk_num)

    for i in range(max_chunk_num):
        dp[0][i].padding = 0
        dp[0][i].last_chunk_end = -1

    for i in range(1, basic_block_num):
        if i >= max_chunk_num and lengths[i] == lengths[i-1]:
            for j in range(0, max_chunk_num):
                dp[i][j].padding = dp[i-1][j].padding
                dp[i][j].last_chunk_end = dp[i-1][j].last_chunk_end
            continue

        dp[i][0].padding = dp[i-1][0].padding + (lengths[i] - lengths[i-1]) * i
        for j in range(1, min(i,max_chunk_num-1)+1):
            dp[i][j].padding = dp[i-1][j-1].padding
            dp[i][j].last_chunk_end = i-1
            if dp[i-1][j].last_chunk_end != -1:
                cum_padding = 0
                for l in range(i-1, dp[i-1][j].last_chunk_end-1, -1):
                    padding = dp[l][j-1].padding + cum_padding
                    if padding < dp[i][j].padding:
                        dp[i][j].padding = padding
                        dp[i][j].last_chunk_end = l
                    cum_padding += lengths[i] - lengths[l]
    if max_padding_ratio is not None:
        max_padding = sum(lengths) * max_padding_ratio
        while dp[basic_block_num-1][max_chunk_num-1].padding < max_padding and max_chunk_num > 1:
            max_chunk_num -= 1
    results = []
    last = basic_block_num-1
    for i in range(max_chunk_num-1, -1, -1):
        results.append(last)
        if i == -1:
            break
        last = dp[last][i].last_chunk_end
    results.reverse()
    return results

cdef check_sorted(lengths):
    """
    Check whether the given list of length is sorted.
    :param lengths: A list of lengths.
    :return: 
    """
    if len(lengths) <= 1:
        return True
    for idx in range(1,len(lengths)):
        if lengths[idx-1] > lengths[idx]:
            return False
    return True

cpdef basic_block_length_chunk(list[int] lengths,
                               int max_chunk_num,
                               max_padding_ratio=None,
                               need_check=True):
    """
    Given the a list of basic block lengths, this function split the list into $n$ chunks, which can result in as less 
        padding as possible. 
    :param lengths: A list of basic block lengths. It should be sorted.
    :param max_chunk_num: How many chunks will these lengths be split into.
    :return: A list of index to describe where the list should be split. There are $n$ sorted elements in result, and 
        the $i$-th element is the index of the last element in the $i$-th chunk. For example, if the result is [1,8,10], 
        then the first chunk should contains the first element, the second chunk consists of the first element to the 8-th
        element and the last chunk contains the 8-th element and 9-th element.
    """
    assert len(lengths) >= max_chunk_num, "The number of basic blocks should be larger than the max_chunk_num"
    if need_check:
        assert check_sorted(lengths), "The given length list is not sorted."
    py_ans = py_solve(lengths, max_chunk_num,max_padding_ratio)
    return py_ans
