import cython
import numpy as np
from numpy import zeros, float32 as REAL
cimport numpy as np

from libc.string cimport memset, memcpy

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

from gensim.models.word2vec_inner cimport bisect_left, random_int32, sscal, REAL_t, EXP_TABLE, our_dot, our_saxpy

DEF MAX_DOCUMENT_LEN = 10000

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef init_a2v_config(Asm2VecConfig *c, model, alpha, learn_tag, learn_tokens, learn_hidden,
                     train_tokens=False, work=None, neu1=None, operands=None, token_vectors=None, word_locks=None,
                     tag_vectors=None, tag_locks=None, dv_count=0):
    # c[0].hs = model.hs
    c[0].negative = model.negative
    c[0].sample = (model.sample != 0)
    c[0].learn_tag = learn_tag
    c[0].learn_tokens = learn_tokens
    c[0].learn_hidden = learn_hidden
    c[0].train_tokens = train_tokens
    c[0].cbow_mean = model.cbow_mean

    if '\0' in model.wv:
        c[0].null_token_index = model.wv.get_index('\0')

    c[0].window = model.window
    c[0].workers = model.workers
    c[0].dv_count = dv_count

    # default vectors, locks from syn0/doctag_syn0
    if token_vectors is None:
        token_vectors = model.wv.vectors
    c[0].token_vectors = <REAL_t *> (np.PyArray_DATA(token_vectors))

    if tag_vectors is None:
        tag_vectors = model.dv.vectors
    c[0].tag_vectors = <REAL_t *> (np.PyArray_DATA(tag_vectors))

    if token_locks is None:
        token_locks = model.wv.vectors_lockf
    c[0].token_locks = <REAL_t *> (np.PyArray_DATA(token_locks))

    if tag_locks is None:
        tag_locks = model.dv.vectors_lockf
    c[0].tag_locks = <REAL_t *> (np.PyArray_DATA(tag_locks))

    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
        work = zeros(model.layer1_size, dtype=REAL)
    c[0].work = <REAL_t *> np.PyArray_DATA(work)

    if neu1 is None:
        neu1 = zeros(model.layer1_size, dtype=REAL)
    c[0].neu1 = <REAL_t *> np.PyArray_DATA(neu1)

    if operands is None:
        operands = zeros(model.wv.vector_size, dtype=REAL)
    c[0].operands = <REAL_t *> np.PyArray_DATA(operands)

    c[0].alpha = alpha
    c[0].layer1_size = model.layer1_size
    c[0].vector_size = model.dv.vector_size

    if c[0].negative:
        c[0].syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        c[0].cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        c[0].cum_table_len = len(model.cum_table)
    if c[0].negative or c[0].sample:
        c[0].next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

def train_document_dm_asm2vec(model, function, alpha, work=None, neu1=None, operands=None,
                              learn_tags=True, learn_tokens=True, learn_hidden=True,
                              token_vectors=None, token_locks=None, tag_vectors=None, tag_locks=None):
    """Update distributed memory model ("PV-DM") by training on a single Function.
    This method implements the DM model with a projection (input) layer that is either the sum or mean of the context
    vectors, depending on the model's `dm_mean` configuration field.

    Called internally from :meth:`~gensim.models.doc2vec.Asm2Vec.train` and
    :meth:`~gensim.models.asm2vec.Asm2vec.infer_vector`.

    Parameters
    ----------
    model : :class:`~gensim.models.asm2vec.Asm2Vec`
        The model to train.
    function : gensim.models.asm2vec.Function
        The input function as a list of instructions to be used for training. Each token in instructions will be looked
        up in the model's vocabulary.
    alpha : float
        Learning rate.
    work : np.ndarray, optional
        Private working memory for each worker.
    neu1 : np.ndarray, optional
        Private working memory for each worker.
    operands: np.ndarray, optional
        Private working memory for each worker.
    learn_tags : bool, optional
        Whether the tag vectors should be updated.
    learn_tokens : bool, optional
        Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
        `learn_words` and `train_words` are set to True.
    learn_hidden : bool, optional
        Whether the weights of the hidden layer will be updated.
    token_vectors : numpy.ndarray, optional
        The vector representation for each word in the vocabulary. If None, these will be retrieved from the model.
    token_locks : numpy.ndarray, optional
        A learning lock factor for each weight in the hidden layer for words, value 0 completely blocks updates,
        a value of 1 allows to update word-vectors.
    tag_vectors : numpy.ndarray, optional
        Vector representations of the tags. If None, these will be retrieved from the model.
    tag_locks : numpy.ndarray, optional
        The lock factors for each tag, same as `word_locks`, but for document-vectors.

    Returns
    -------
    int
        Number of words in the input document that were actually used for training.

    """
    cdef Asm2VecConfig c

    cdef REAL_t  inv_count_opnd, inv_count
    cdef int i, j, k, m, o, z
    cdef int t, predict_word_index
    cdef long result = 0

    init_a2v_config(&c, model, alpha, learn_tags, learn_tokens, learn_hidden, train_tokens=False,
                    work=work, neu1=neu1, operands=operands, token_vectors=token_vectors, token_locks=token_locks,
                    tag_vectors=tag_vectors, tag_locks=tag_locks)

    # NOTE: Set the number of function's tags and instructions
    c.function_len = <int> min(MAX_DOCUMENT_LEN, len(function.instructions))

    operand_cnt = 0
    for i, instruction in enumerate(function.instructions):

        # 10,000
        if i >= MAX_DOCUMENT_LEN:
            break

        # 50,000
        if operand_cnt >= MAX_TOKENS_LEN:
            break

        # INSTRUCTION.OPERATOR
        operator_index = model.wv.key_to_index.get(instruction.operator, None)
        if operator_index:
            c.operators_idx[i] = operator_index
        else:
            c.operators_idx[i] = c.null_token_index

        # INSTRUCTION.OPERANDS
        # Keep track of how many operands per instruction
        opnd_cnt_i = 0
        # ... and of the starting offset for the current instruction
        opnd_offset_i = operand_cnt
        for operand in instruction.operands:

            operand_index = model.wv.key_to_index.get(operand, None)
            if operand_index:
                c.operands_idx[operand_cnt] = operand_index
                operand_cnt += 1
                opnd_cnt_i += 1

        c.operands_len[i] = opnd_cnt_i
        c.operands_offset[i] = opnd_offset_i

    # NOTE: Save the indexes for the function's tags in the c struct.
    c.tag_index = model.dv.get_index(function.tag, None)

    # release GIL & train on the document
    with nogil:

        # NOTE: "i" is the index for the target instruction in the document
        # NOTE: "j" and "k" are the bounds of the window.
        for i in range(c.function_len):
            j = i - c.window
            if j < 0:
                j = 0
            k = i + c.window + 1
            if k > c.function_len:
                k = c.function_len

            # Define the predict_word_index
            # i is the target instruction
            for t in range(c.operands_len[i] + 1):
                if t == c.operands_len[i]:
                    # the target to predict is the operator
                    predict_word_index = c.operators_idx[i]
                else:
                    # the target to predict is one of the operands
                    predict_word_index = c.operands_idx[c.operands_offset[i] + t]

                # compose l1 (in _neu1) & clear _work
                memset(c.neu1, 0, c.layer1_size * cython.sizeof(REAL_t))

                # NOTE: "m" is the index of the instruction in the window
                for m in range(j, k):
                    if m == i:
                        # target instruction, skip it
                        continue
                    else:
                        # summarize the embeddings of operators in context
                        our_saxpy(&c.vector_size, &ONEF, &c.token_vectors[c.operators_idx[m] * c.vector_size], &ONE,
                                  c.neu1, &ONE)

                        # average the embeddings of operands in current instruction
                        memset(c.operands, 0, c.vector_size * cython.sizeof(REAL_t))
                        inv_count_opnd = <REAL_t> 1.0

                        for o in range(c.operands_len[m]):
                            z = c.operands_idx[c.operands_offset[m] + o]
                            our_saxpy(&c.vector_size, &ONEF, &c.token_vectors[z * c.vector_size], &ONE, c.operands,
                                      &ONE)

                        if c.operands_len[m] > 0:
                            inv_count_opnd = ONEF / c.operands_len[m]

                        sscal(&c.vector_size, &inv_count_opnd, c.operands,
                              &ONE)  # (does this need BLAS-variants like saxpy?)

                        # summarize the embeddings of operators in context
                        our_saxpy(&c.vector_size, &ONEF, c.operands, &ONE, &c.neu1[c.vector_size], &ONE)
                # add the embedding of function name to neu1
                our_saxpy(&c.layer1_size, &ONEF, &c.tag_vectors[c.tag_index * c.layer1_size], &ONE, c.neu1, &ONE)
                # calculate the average embeddings
                inv_count = ONEF / (k-j)
                sscal(&c.layer1_size, &inv_count_opnd, c.neu1, &ONE)

                # vector work is used to accumulate l1 error
                memset(c.work, 0, c.layer1_size * cython.sizeof(REAL_t))

                # I don't know what this function do, but it seems to calculate the gradient and save it in c.work
                c.next_random = fast_document_dm_neg(c.negative, c.cum_table, c.cum_table_len, c.next_random,
                                                     c.neu1, c.syn1neg, predict_word_index, c.alpha, c.work,
                                                     c.layer1_size,
                                                     c.learn_hidden)

                sscal(&c.layer1_size, inv_count, c.work, &ONE)

                # apply accumulated error in work
                # NOTE: Check when doctag_locks are set.
                if c.learn_tag:
                    our_saxpy(&c.layer1_size, &c.tag_locks[c.tag_index], c.work,
                              &ONE, &c.tag_vectors[c.tag_index * c.layer1_size], &ONE)

                # NOTE: learn the tags too
                if c.learn_tokens:
                    for m in range(j, k):
                        if m == i:
                            continue
                        else:
                            # OPERATOR
                            our_saxpy(&c.vector_size, &c.token_locks[c.operators_idx[m]], c.work, &ONE,
                                      &c.token_vectors[c.operators_idx[m] * c.vector_size], &ONE)

                            # OPERANDS
                            memset(c.operands, 0, c.vector_size * cython.sizeof(REAL_t))
                            inv_count_opnd = <REAL_t> 1.0
                            # NOTE: copy the error relative the operands to c.tmp_opnds
                            our_saxpy(&c.vector_size, &ONEF, &c.work[c.vector_size],
                                      &ONE, c.operands, &ONE)

                            if c.operands_len[m] > 0:
                                inv_count_opnd = ONEF / c.operands_len[m]

                            sscal(&c.vector_size, &inv_count_opnd, c.operands,
                                  &ONE)  # (does this need BLAS-variants like saxpy?)

                            for o in range(c.operands_len[m]):
                                z = c.operands_idx[c.operands_offset[m] + o]
                                our_saxpy(&c.vector_size, &c.token_locks[z], c.operands, &ONE,
                                          &c.token_vectors[z * c.vector_size], &ONE)

    return result
