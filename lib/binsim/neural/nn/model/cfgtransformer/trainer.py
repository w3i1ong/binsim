import torch
from torch import nn
from torch.utils.data import DataLoader
from .schedule import ScheduledOptim
from .cfgtransformer import CFGTransformer, InstructionEmbedding
from .instruction import BatchedFunctionSeq
from torch.optim import Adam
from tqdm import tqdm
from typing import Tuple
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import numpy as np

class CFGBERTLM(nn.Module):
    def __init__(self, cfg_bert: CFGTransformer, vocab_size: int):
        super().__init__()
        self.cfg_bert = cfg_bert
        self.ins_embedding = InstructionEmbedding(vocab_size, cfg_bert.hidden)

    def forward(self, instructions: BatchedFunctionSeq)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        :param instructions: The input instruction sequence.
        :return:
        """
        candidate_embedding = self.ins_embedding(instructions.masked_predict)
        predict_embedding:torch.Tensor = self.cfg_bert(instructions)
        masked_embedding = predict_embedding.reshape(predict_embedding.shape[0] * predict_embedding.shape[1],
                                                    predict_embedding.shape[-1],
                                                    1)
        masked_embedding = masked_embedding[instructions.masked_predict_idx]
        return predict_embedding, masked_embedding, candidate_embedding

class CFGBERTTrainer:
    def __init__(self,
                 cfg_bert: CFGTransformer,
                 vocab_size: int,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 lr = 1e-3,
                 betas = (0.9, 0.999),
                 weight_decay = 0.01,
                 warmup_steps = 10000,
                 with_cuda = True,
                 log_interval = 100,):
        """
        :param cfg_bert: The CFG-BERT model to train.
        :param vocab_size: Total number of tokens in the dictionary.
        :param train_loader: Training data loader.
        :param valid_loader: Validation data loader.
        :param lr: Learning rate of the Adam optimizer.
        :param betas: betas of the Adam optimizer.
        :param weight_decay: Weight decay of the Adam optimizer.
        :param warmup_steps: Number of warmup steps for the learning rate scheduler.
        :param with_cuda: Whether to use GPU.
        :param log_interval: Logging interval of the training process.
        """
        if with_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.bert = cfg_bert
        self.language_model = CFGBERTLM(cfg_bert, vocab_size).to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = Adam(self.language_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.log_interval = log_interval
        self.grad_scaler = GradScaler()
        self.optim_schedule = ScheduledOptim(self.optimizer, self.grad_scaler, self.bert.hidden, warmup_steps)
        print("Total Parameters:", sum([p.nelement() for p in self.language_model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_loader)

    def iteration(self, epoch, data_loader, train=True):
        stage = "Train" if train else "Valid"
        data_process_bar = tqdm(enumerate(data_loader),
                                desc=f"{stage} epoch {epoch:02d}",
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}")
        loss_list, accuracy_list = [], []

        self.optim_schedule.zero_grad()
        for iter_num, masked_sequence  in data_process_bar:
            masked_sequence.to(self.device)
            with autocast():
                encoder_embedding, masked_embedding, candidate_embedding = self.language_model(masked_sequence)
                predict:torch.Tensor = torch.squeeze(candidate_embedding @ masked_embedding)
                labels = torch.zeros(predict.shape[0], dtype=torch.long, device=predict.device)
                mask_loss = self.criterion(predict, labels)
                loss = mask_loss
            if train:
                self.optim_schedule.backward(loss)
                self.optim_schedule.step_and_update_lr()
                self.optim_schedule.zero_grad()

            loss_list.append(loss.cpu().detach().item())
            predict_label = torch.argmax(predict, dim=1).detach().cpu().numpy()
            accuracy_list.append(accuracy_score(np.zeros_like(predict_label), predict_label))
            # delete no longer used variables to save memory
            del loss, mask_loss, labels, predict, masked_embedding, candidate_embedding, encoder_embedding

            if (iter_num + 1) % self.log_interval == 0 or iter_num == len(data_loader) - 1:
                avg_loss = sum(loss_list) / len(loss_list)
                avg_accuracy = sum(accuracy_list) / len(accuracy_list)
                data_process_bar.write(f"Epoch: {epoch:02d}, Iter: {iter_num:06d}, Avg Loss: {avg_loss:.5f},"
                                       f" Avg acc of MLM: {avg_accuracy:.5f}")
                loss_list, accuracy_list = [], []


    @torch.no_grad()
    def valid(self, epoch):
        self.iteration(epoch, self.valid_loader, train=False)

    def save(self, epoch, path="output/bert_trained.model"):
        output_model_file = f'{path}.{epoch:02d}.bin'
        torch.save(self.bert.cpu(), output_model_file)
        self.bert.to(self.device)
        return output_model_file
