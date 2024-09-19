import numpy as np
cimport numpy as np

ctypedef np.float32_t REAL_t
DEF MAX_DOCUMENT_LEN = 10000
DEF MAX_TOKENS_LEN = MAX_DOCUMENT_LEN * 6
DEF MAX_TOKENS_LEN = MAX_DOCUMENT_LEN * 6


cdef struct Asm2VecConfig:
    # int hs,
    int negative, sample, learn_tag, learn_tokens, learn_hidden, train_words, cbow_mean
    int function_len, window, null_token_index, workers, dv_count

    REAL_t *token_vectors
    REAL_t *token_locks

    REAL_t *tag_vectors
    REAL_t *tag_locks

    REAL_t *work
    REAL_t *neu1
    REAL_t *operands

    REAL_t alpha
    int layer1_size, vector_size

    np.uint32_t tag_index
    int codelens[MAX_DOCUMENT_LEN]
    np.uint32_t indexes[MAX_DOCUMENT_LEN]
    np.uint32_t window_indexes[MAX_DOCUMENT_LEN]

    # For instructions
    np.uint32_t operators_idx[MAX_DOCUMENT_LEN]
    np.uint32_t operands_idx[MAX_TOKENS_LEN]
    np.uint32_t operands_len[MAX_DOCUMENT_LEN]
    np.uint32_t operands_offset[MAX_DOCUMENT_LEN]

    # For hierarchical softmax
    # REAL_t *syn1
    # np.uint32_t *points[MAX_DOCUMENT_LEN]
    # np.uint8_t *codes[MAX_DOCUMENT_LEN]

    # For negative sampling
    REAL_t *syn1neg
    np.uint32_t *cum_table
    unsigned long long cum_table_len, next_random

cdef init_a2v_config(Asm2VecConfig *c, model, alpha, learn_doctags, learn_words, learn_hidden, train_words= *, work= *,
                     neu1= *, opnds= *, word_vectors= *, word_locks= *, doctag_vectors= *, doctag_locks= *,
                     docvecs_count= *)
