import shutil
import time
import torch
import pickle
import logging
from tqdm import tqdm
from torch import Tensor
from torch.optim import Adam
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Tuple, List, Iterable, Set, Union, Any
from binsim.neural.nn.metric import *
from binsim.neural.nn.base.model import GraphEmbeddingModelBase, GraphMatchingModelBase
from binsim.utils import get_loss_by_name
from binsim.neural.utils import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.nn.globals.siamese import *

logger = logging.getLogger('Siamese')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'))
logger.addHandler(handler)

class EmbeddingQueue:
    def __init__(self, max_samples):
        self._capacity = max_samples
        self._data = None
        self._full = False
        self._index = 0

    def put(self, sample):
        if self._data is None:
            self._data = torch.zeros((self._capacity, sample.shape[1]), device=sample.device, dtype=sample.dtype)
        if self._index + sample.size(0) >= self._capacity:
            self._full = True
        assert len(sample) <= self._capacity, "The sample size is larger than the capacity of the queue."
        if self._index + sample.size(0) <= self._capacity:
            self._data[self._index:self._index + sample.size(0)] = sample
        else:
            left = self._capacity - self._index
            self._data[self._index:] = sample[:left]
            self._data[:sample.size(0) - left] = sample[left:]
        self._index = (self._index + sample.size(0)) % self._capacity


    def get_tensors(self):
        if self._full:
            return self._data
        else:
            return self._data[:self._index]

    def empty(self):
        return not self._full and self._index == 0


class Siamese(object):
    def __init__(self,
                 model: GraphEmbeddingModelBase,
                 optimizer=Adam,
                 device='cpu',
                 sample_format=SiameseSampleFormat.Pair,
                 sampler=None,
                 momentum_model=None):
        """
        Initialize models, loss, sample method and optimizer for our siamese.
        :param model: the first model. It takes a batch of samples as input,
            and output related embeddings.
        :param optimizer: any optimizer in torch.optim.
        :return:
        """
        super(Siamese, self).__init__()
        self._device = device
        self._model = model
        self._optimizer = optimizer
        self._sample_format = sample_format
        self._sampler = sampler
        self._momentum_model = momentum_model
        self._target_queue = None

    @property
    def model(self) -> Union[GraphEmbeddingModelBase, GraphMatchingModelBase]:
        return self._model

    @property
    def use_momentum_model(self):
        return self._momentum_model is not None

    @model.setter
    def model(self, model: Union[GraphEmbeddingModelBase, GraphMatchingModelBase]):
        self._model = model.to(self._device)

    @property
    def device(self):
        return self._device

    def _generate_embedding(self, samples) -> Tensor:
        """
        Generate embedding for samples with the model.
        :param samples: A batch of samples.
        :return: The embedding of samples.
        """
        if isinstance(samples, (list, tuple)):
            samples = list(sample.to(self.device) if hasattr(sample, 'to') else sample for sample in samples)
            embedding = self.model.generate_embedding(*samples)
        else:
            samples = samples.to(self._device)
            embedding = self.model.generate_embedding(samples)
        return embedding

    def parameters(self) -> Iterable:
        """
        Get the parameters of the siamese model and loss. As some loss functions may have parameters to train, we should
        take them into consideration.
        :return:
        """
        result = self._model.parameters()
        return result

    def forward(self, samples, labels, sample_ids, loss_func) -> torch.Tensor:
        labels = labels.to(self._device)
        if self._momentum_model is not None:
            query, target = samples
            with torch.no_grad():
                target = tuple(sample.to(self._device) if hasattr(sample, 'to') else sample for sample in target)
                target_embedding = self._momentum_model(target)
            query = tuple(sample.to(self._device) if hasattr(sample, 'to') else sample for sample in query)
            query_embedding = self.model(query)
            if not self._target_queue.empty():
                extra_target_embeddings = self._target_queue.get_tensors()
                target_embeddings = torch.cat([target_embedding, extra_target_embeddings], dim=0)
            else:
                target_embeddings = target_embedding
            target_ids = torch.arange(0, len(query_embedding), device=self._device)
            self._target_queue.put(target_embedding)
            return loss_func((query_embedding, target_embeddings), target_ids,
                             sample_ids=None,
                             pair_sim_func=self.model.similarity,
                             pairwise_sim_func=self.model.pairwise_similarity)
        else:
            samples = tuple(sample.to(self._device) if hasattr(sample, 'to') else sample for sample in samples)
            embeddings = self.model.generate_embedding(*samples)
            sample_format = self._sample_format
            if self._sampler is not None:
                embeddings, sample_ids, labels, sample_format = (
                    self._sampler(embeddings, sample_ids, labels, self.model.pairwise_similarity, self._sample_format,
                                  self.model.distance_metric))
            loss_func.check(sample_format, self.model.distance_metric)
            return loss_func(embeddings, sample_ids=sample_ids, labels=labels, pair_sim_func=self.model.similarity,
                             pairwise_sim_func=self.model.pairwise_similarity)

    def train(self,
              train_data: RandomSamplePairDatasetBase,
              record_dir: str,
              search_data: List[SampleDatasetBase],
              epoch: int = 50,
              val_interval=5,
              choice_metric: SiameseMetric = SiameseMetric.nDCG(10),
              metrics: Set[SiameseMetric] = None,
              backward_steps=1,
              lr: float = 0.001,
              momentum_model_update_rate: float = 0.001,
              loss_func_name = None,
              loss_func_kwargs: dict = None,
              optimizer_kwargs: dict = None,
              ignore_first=False,
              num_workers=0,
              train_batch_size=32,
              eval_search_batch_size=32,
              queue_max_size=100,
              lr_update_epoch=None,
              lr_update_scale=0.9
              ) -> None:
        """
        Train the siamese model with train_dataloader and val_dataloader, and save the model with the best metric.
        :param train_data: The dataloader for training data.
        :param record_dir:  The directory to record the trend of metrics.
        :param search_data: The dataloader for searching.
        :param epoch:  The number of epoch to train.
        :param val_interval: How many epochs to validate once.
        :param choice_metric: The metric to choose the best model.
        :param metrics: The metrics to record, can be a subset of SUPPORTED_METRIC.
        :param backward_steps: How many steps to backward once
        :param lr: The learning rate.
        :param momentum_model_update_rate: The update rate of momentum model.
        :param loss_func_name: The name of loss function.
        :param loss_func_kwargs: The kwargs to be passed to loss function.
        :param optimizer_kwargs: Extra parameters passed to optimizer.
        :param ignore_first: Whether the first search result should be ignored.
        :param num_workers: Number of workers for Dataloaders to load samples.
        :param train_batch_size: The batch_size used to load training samples.
        :param queue_max_size: The max size of the queue used in momentum model.
        :param eval_search_batch_size: The batch size used in search evaluation.
        :param lr_update_epoch: If set, the learning rate will be updated, when the loss doesn't change for
            lr_update_epoch epochs.
        :param lr_update_scale: If lr_update_epoch is set, this argument will be used to update the learning rate by
            multiplication.
        :return:
        """
        logger.info(f"Start training with record_dir: {record_dir}")
        logger.info(f"Distance metric: {self.model.distance_metric}")
        logger.info(f"Sample format: {self._sample_format}")
        logger.info(f"Sampler: {self._sampler}")
        logger.info(f"Loss function: {loss_func_name}")
        # initialize optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = self._optimizer(self.parameters(), lr=lr, **optimizer_kwargs)
        optimizer.zero_grad()
        # initialize loss function
        if loss_func_kwargs is None:
            loss_func_kwargs = {}
        loss_func = get_loss_by_name(loss_func_name, **loss_func_kwargs)
        if self._momentum_model is not None:
            self._prepare_momentum_model()
            self._target_queue = EmbeddingQueue(queue_max_size)

        # initialize metric recorder
        metrics = set(metrics)
        metrics.add(choice_metric)
        loss_list, metric_record = [], {}
        train_time_list = []
        for metric in metrics:
            metric_record[metric] = []
        best_metric = None
        best_metric_name = choice_metric

        # initialize model
        self.model.sample_format = train_data.sample_format = self._sample_format

        # initialize dataloader
        train_dataloader = DataLoader(train_data, batch_size=train_batch_size // backward_steps, shuffle=True,
                                      collate_fn=train_data.collate_fn, num_workers=num_workers, drop_last=True)
        # For classification, we need use randomly generated sample pairs
        query_data, target_data = search_data
        query_dataloader = DataLoader(query_data, batch_size=eval_search_batch_size, shuffle=False,
                                      collate_fn=query_data.collate_fn_with_name, num_workers=num_workers)
        target_dataloader = DataLoader(target_data, batch_size=eval_search_batch_size, shuffle=False,
                                       collate_fn=target_data.collate_fn_with_name, num_workers=num_workers)
        search_loader = (query_dataloader, target_dataloader)

        loss_not_decrease, min_loss = 0, 1e10
        # train loop
        for e in range(epoch):
            # set processbar
            self.model.train()
            processbar = tqdm(train_dataloader)
            processbar.set_description_str(f'[epoch: {e:2}]')

            batch_loss_list = []
            # train iteration
            train_time_start = time.time()
            for batch_idx, (samples, sample_ids, labels) in enumerate(processbar):
                # forward and backward
                loss = self.forward(samples, labels, sample_ids, loss_func=loss_func)
                loss.backward()
                if (batch_idx + 1) % backward_steps == 0 or batch_idx == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                loss_value = loss.detach().cpu().item()
                processbar.set_postfix_str(f"loss:{loss_value:.4f}")
                # record loss
                batch_loss_list.append(loss_value)

                if self._momentum_model is not None:
                    # update momentum model
                    with torch.no_grad():
                        for param, momentum_param in zip(self.model.parameters(), self._momentum_model.parameters()):
                            momentum_param.data = momentum_param.data * (1 - momentum_model_update_rate) + \
                                                  param.data * momentum_model_update_rate

            torch.cuda.synchronize()
            average_train_time = (time.time()-train_time_start)/len(train_data)
            train_time_list.append(average_train_time)
            batch_loss = sum(batch_loss_list) / len(batch_loss_list)
            loss_list.append(batch_loss)
            logger.info(f'Loss of epoch {e:002d}: {batch_loss:3.4f}')
            logger.info(f'Average of train time {e:002d}: {average_train_time*1000:3.4f}ms({train_data.sample_format})')
            if lr_update_epoch is not None:
                if batch_loss < min_loss:
                    min_loss = batch_loss
                    loss_not_decrease = 0
                else:
                    loss_not_decrease += 1
                if loss_not_decrease >= lr_update_epoch:
                    loss_not_decrease = 0
                    lr = lr * lr_update_scale
                    for parameter_group in optimizer.param_groups:
                        parameter_group['lr'] = lr
                    min_loss = batch_loss
                    logger.info(f"The lr has been update to {lr:.6f}, "
                                f"as the loss has not decreased for {lr_update_epoch} epochs.")
                else:
                    logger.info(f"Current min_loss: {min_loss:.4f}, it has been not decreased for {loss_not_decrease} epochs.")
            # evaluate every val_interval epoch
            if (e + 1) % val_interval == 0:
                eval_results = self.test(search_loader=search_loader,
                                         metrics=metrics, ignore_first=ignore_first)
                logger.info(f'Evaluation results of epoch {e:002d}: ')
                for metric_name, metric_value in eval_results.items():
                    logger.info(f'{metric_name}: {metric_value}')
                for metric_name, metric_value in eval_results.items():
                    metric_record[metric_name].append(metric_value)

                if best_metric_name in metric_record and \
                        (best_metric is None or metric_record[best_metric_name][-1] > best_metric):
                    best_metric = metric_record[best_metric_name][-1]
                    logger.info(f'Get better {best_metric_name}: {best_metric}.')
                    self.save_model(f'{record_dir}/best.pkl')
                    shutil.copy(f'{record_dir}/best.pkl', f'{record_dir}/latest.pkl')
                else:
                    self.save_model(f'{record_dir}/latest.pkl')
        self.save_model(f'{record_dir}/last.pkl')

        # save recorded metrics
        with open(f'{record_dir}/train/loss.pkl', 'wb') as f:
            pickle.dump(loss_list, f)
        with open(f'{record_dir}/train/train_time.pkl', 'wb') as f:
            pickle.dump(loss_list, f)
        with open(f'{record_dir}/train/metric.pkl', 'wb') as f:
            pickle.dump(metric_record, f)

    def _prepare_momentum_model(self):
        # copy parameters from model to momentum_model
        self._momentum_model.load_state_dict(self._model.state_dict())
        for param in self._momentum_model.parameters():
            param.requires_grad = False
        self._momentum_model.eval()

    def test(self, search_loader: Tuple[DataLoader, DataLoader],
             metrics: Set[SiameseMetric],
             verbose = True,
             ignore_first=False) -> dict[SiameseMetric, float]:
        self.model.eval()
        search_metrics = {metric for metric in metrics if metric.is_search_metric()}

        # todo: for GraphMatchingModel, we should use another method to evaluate search,
        #   as for GraphMatchingModel, we should use the original graphs to calculate the similarity.
        sample_loader1, sample_loader2 = search_loader
        if isinstance(self.model, GraphEmbeddingModelBase):
            search_result = self.evaluate_search_for_embedding_model(sample_loader1,
                                                                     sample_loader2,
                                                                     search_metrics,
                                                                     ignore_first=ignore_first,
                                                                     verbose=verbose)
        else:
            # raise RuntimeError("Search evaluation for GraphMatchingModel is too slow, so we just comment it.")
            search_result = self.evaluate_search_for_matching_model(sample_loader1,
                                                                    sample_loader2,
                                                                    search_metrics,
                                                                    ignore_first=ignore_first)
        return search_result

    def save_model(self, filename):
        self.model.save(filename)

    def generate_embeddings_for_search_eval(self, samples: DataLoader, verbose=True):
        embeddings, ids, names, tags = [], [], [], []
        if verbose:
            samples = tqdm(samples)
        for batched_samples, batched_ids, batched_names, batched_tags in samples:
            batched_embeddings = self._generate_embedding(batched_samples)
            embeddings.append(batched_embeddings.cpu())
            ids.append(batched_ids.cpu())
            names.extend(batched_names)
            tags.extend(batched_tags)
        return torch.cat(embeddings), torch.cat(ids), names, tags

    @torch.no_grad()
    def evaluate_search_for_embedding_model(self,
                                            samples1: DataLoader,
                                            samples2: DataLoader,
                                            metrics: Set[SiameseMetric],
                                            ignore_first=False,
                                            batch_size=32,
                                            verbose=True,
                                            *, debug=False) -> Union[dict, Tuple[dict, List]]:
        self.model.eval()
        # generate embeddings for samples
        embeddings1, ids1, names1, tags1 = self.generate_embeddings_for_search_eval(samples1, verbose=verbose)
        embeddings2, ids2, names2, tags2 = self.generate_embeddings_for_search_eval(samples2, verbose=verbose)
        # calculate search metrics
        # 1. calculate search rank
        search_result, answer_num = search(embeddings1, ids1,
                                           embeddings2, ids2,
                                           pair_sim_func=self.model.pairwise_similarity_for_search,
                                           device=self._device,
                                           top_k=200 + ignore_first,
                                           verbose=verbose,
                                           batch_size=batch_size)
        ids1 = ids1.to(self._device).reshape([-1, 1])
        ids2 = ids2.to(self._device)
        search_result_correct = (ids1 == ids2[search_result]).float()
        metrics = self.calculate_search_metrics(search_result_correct, answer_num, metrics, ignore_first=ignore_first)

        if not debug:
            return metrics
        # 1. analysis those samples with wrong search result @1
        ids1 = ids1.reshape([-1])
        wrong_index = (search_result_correct[:,0] == 0)
        wrong_ids1, wrong_ids2 = ids1[wrong_index], ids2[search_result[wrong_index]]
        _, old_idx1 = torch.sort(ids1)
        _, old_idx2 = torch.sort(ids2)
        tags1 = [tags1[idx] for idx in old_idx1.cpu().numpy()]
        tags2 = [tags2[idx] for idx in old_idx2.cpu().numpy()]

        wrong_results_name = []
        for query_idx, row in zip(wrong_ids1.cpu().numpy(), wrong_ids2.cpu().numpy()):
            line = []
            for idx in row:
                line.append(tags2[idx])
            wrong_results_name.append((tags1[query_idx],line))

        diff_distribution = self.analysis_tag_difference_distribution(search_result, ids1, ids2, tags1, tags2)
        return metrics, diff_distribution

    def analysis_tag_difference_distribution(self, search_results: torch.Tensor, src_ids, dst_ids,
                                             src_tags: List[Any], dst_tags: List[Any]) -> List:
        def tag_distance(src_tag, dst_tag):
            _, arch1, _, _, opt_level1 = src_tag
            _, arch2, _, _, opt_level2 = dst_tag
            result = 0
            if arch1 != arch2:
                result += 1
            opt_level1 = int(opt_level1[1:])
            opt_level2 = int(opt_level2[1:])
            return result + abs(opt_level1 - opt_level2)

        statistics = defaultdict(lambda: list())
        search_results = search_results.cpu().numpy()
        src_ids = src_ids.cpu().numpy()
        dst_ids = dst_ids.cpu().numpy()
        for src_id, search_result in zip(src_ids, search_results):
            src_id = src_id.item()
            src_tag = src_tags[src_id]
            count = 0
            for rank, result_idx in enumerate(search_result):
                if dst_ids[result_idx] != src_id:
                    count += 1
                    continue
                result_tag = dst_tags[result_idx]
                statistics[tag_distance(src_tag, result_tag)].append(count)
        pass

    def evaluate_search_for_matching_model(self,
                                           samples1: DataLoader,
                                           samples2: DataLoader,
                                           metrics: Set[SiameseMetric],
                                           ignore_first=False,
                                           batch_size=32) -> dict[SiameseMetric, float]:
        self.model.eval()
        search_result = []
        answer_num = []
        target_id_freq = {}
        for query_samples, query_ids, query_names in samples1:
            pairwise_similarity = []
            target_ids = []
            # calculate pairwise similarity between query and search
            for search_samples, search_id, search_names in tqdm(samples2):
                similarity = self.model.pairwise_similarity_between_original(query_samples, search_samples)
                pairwise_similarity.append(similarity)
                target_ids.append(search_id)
            pairwise_similarity = torch.cat(pairwise_similarity, dim=1)
            target_ids = torch.cat(target_ids)

            if not target_id_freq:
                for target_id in target_ids.cpu().numpy():
                    target_id_freq[target_id] = target_id_freq.get(target_id, 0) + 1

            # calculate the top-100 similarity, and search result
            _, result_index = torch.topk(pairwise_similarity, dim=1, k=100 + ignore_first)
            result_id = target_ids[result_index]
            search_result = (result_id == query_ids.view(-1, 1)).float()

            # calculate the answer number
            query_answer_num = torch.tensor([target_id_freq.get(query_id, 0) for query_id in query_ids.cpu().numpy()])
            answer_num.append(query_answer_num)

        search_result = torch.cat(search_result, dim=0)
        answer_num = torch.cat(answer_num, dim=0)
        return self.calculate_search_metrics(search_result, answer_num, metrics, ignore_first=ignore_first)

    @staticmethod
    def calculate_search_metrics(search_result, answer_num, metrics: Set[SiameseMetric], ignore_first=False):
        results = {}
        for metric in metrics:
            k = metric.top_value
            match metric.name:
                case 'mrr':
                    results[metric] = calculate_mrr(search_result, answer_num, k, ignore_first=ignore_first)
                case 'hit':
                    results[metric] = calculate_topk_hit(search_result, answer_num, k, ignore_first=ignore_first)
                case 'recall':
                    results[metric] = calculate_topk_recall(search_result, answer_num, k, ignore_first=ignore_first)
                case 'precision':
                    results[metric] = calculate_topk_precision(search_result, answer_num, k, ignore_first=ignore_first)
                case 'ndcg':
                    results[metric] = calculate_topk_nDCG(search_result, answer_num, k, ignore_first=ignore_first)
                case _:
                    raise ValueError(f"Unsupported metric: {metric}")
        return results
