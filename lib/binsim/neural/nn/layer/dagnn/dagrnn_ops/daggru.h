#ifndef DAGGRU_H
#define DAGGRU_H
#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

void message_passing_forward(Tensor hidden, Tensor last_hidden, Tensor edge_batch, Tensor edge_batch_index);
void fused_gru_partial_forward(Tensor last_hidden, Tensor hidden, Tensor gru_wx, Tensor gru_wh, Tensor gru_bias,
                               long node_base, long node_num);
void fused_gru_partial_backward(Tensor grad_hidden, Tensor grad_last_hidden, Tensor last_hidden,
                                Tensor gru_wx, Tensor gru_wh, Tensor gru_bias,
                                long node_base, long node_num);
void fused_lstm_partial_forward(Tensor last_cell, Tensor last_hidden, Tensor cell, Tensor hidden,
                                Tensor i, Tensor f, Tensor g, Tensor o);
void fused_lstm_partial_backward(Tensor grad_cell, Tensor grad_hidden, Tensor grad_last_cell, Tensor grad_last_hidden,
                                Tensor last_cell, Tensor last_hidden, Tensor i, Tensor f, Tensor g, Tensor o);
#endif
