import cython
import numpy as np
cimport numpy as cnp
from hashlib import sha256
from scipy.sparse.csgraph import floyd_warshall

# todo: make the hash function much more robust, precise and efficient.
cpdef compute_function_hash(function):
    """
    Compute the hash of a function. This function implements the hash function proposed described in the paper
    [safe](https://arxiv.org/pdf/1811.05296.pdf) to avoid multiple inclusion of common functions and libraries.
    In concrete, we first replace the immediate operands and the memory operands with a special symbol(namely
    [IMM] and [MEM]), then we compute the hash of the instruction sequence as the hash of the function.
    :param function: The function to be hashed.
    :return: The hash of the function.
    """
    function_instructions = []
    for block in function:
        for tokens, _ in block:
            ins_tokens = []
            for token in tokens:
                if token.type in {token.type.TextToken,token.type.OperandSeparatorToken}:
                    ins_tokens.append(token.text.strip())
                elif token.type == token.type.IntegerToken:
                    ins_tokens.append("[IMM]")
                elif token.type in { token.type.PossibleAddressToken}:
                    ins_tokens.append("[ADDR]")
                elif token.type == token.type.CodeRelativeAddressToken:
                    ins_tokens.append("[REL]")
                elif token.type == token.type.FloatingPointToken:
                    ins_tokens.append("[FLOAT]")
                elif token.type == token.type.CommentToken:
                    pass
                else:
                     ins_tokens.append(token.text)
            function_instructions.append(" ".join(ins_tokens))
    return sha256("".join(function_instructions).encode()).hexdigest()


cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def convert_bb_dis_matrix_to_ins_dis_matrix(cnp.ndarray[double, ndim=2] bb_distance_matrix,
                                            cnp.ndarray[int,ndim=1] bb_ins_num,
                                            cnp.ndarray[int,ndim=1] ins_idx_base):
    cdef int ins_total_num = sum(bb_ins_num)
    cdef int row, col, delta_row, delta_col, row_base, col_base
    cdef double distance
    cdef int [:,:] distance_matrix_view
    cdef double [:, :] bb_distance_matrix_view
    cdef int [:] bb_ins_num_view = bb_ins_num
    cdef int [:] ins_idx_base_view = ins_idx_base

    bb_distance_matrix_view = bb_distance_matrix

    distance_matrix = np.zeros((ins_total_num, ins_total_num), dtype=np.int32)
    distance_matrix_view = distance_matrix

    for row in range(bb_distance_matrix_view.shape[0]):
        for col in range(bb_distance_matrix_view.shape[1]):
            distance = bb_distance_matrix_view[row, col]
            if distance == np.inf:
                continue
            row_base = ins_idx_base_view[row]
            col_base = ins_idx_base_view[col]
            for delta_row in range(bb_ins_num_view[row]):
                for delta_col in range(bb_ins_num_view[col]):
                    distance_matrix_view[row_base+delta_row, col_base + delta_col] = delta_col - delta_row + int(distance)
    return distance_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef control_reachable_distance_matrix(features, adj_list):
    cdef int bb_num = len(features)
    cdef int[:,:] bb_adj_list_view
    cdef int[:,] bb_ins_num_view
    cdef int[:,] ins_idx_base_view
    cdef int i, j

    bb_ins_num = np.zeros((bb_num,), dtype=np.int32)
    bb_ins_num_view = bb_ins_num
    for i in range(bb_num):
        bb_ins_num_view[i] = len(features[i])

    # the index base of each basic block
    ins_idx_base = np.zeros((bb_num,), dtype=np.int32)
    ins_idx_base_view = ins_idx_base
    for i in range(bb_num-1):
        ins_idx_base_view[i+1] = ins_idx_base_view[i] + bb_ins_num_view[i]

    # calculate the basic block reachable matrix
    bb_adj_list = np.zeros((bb_num, bb_num), dtype=np.int32)
    bb_adj_list_view = bb_adj_list

    for i in range(len(adj_list)):
        src, dst = adj_list[i]
        bb_adj_list_view[src, dst] = bb_ins_num_view[src]

    bb_distance_matrix = floyd_warshall(bb_adj_list)
    distance_matrix = convert_bb_dis_matrix_to_ins_dis_matrix(bb_distance_matrix, bb_ins_num, ins_idx_base)
    return distance_matrix
