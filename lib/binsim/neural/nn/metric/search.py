import torch
import tqdm
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from tqdm import tqdm

@torch.no_grad()
def search(src_embeddings: torch.Tensor,
           src_ids: torch.Tensor,
           target_embedding: torch.Tensor,
           target_ids: torch.Tensor,
           pair_sim_func,
           device='cpu',
           top_k=100,
           verbose=True,
           batch_size=32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Search each src embedding in target_embeddings.
    :param src_embeddings: The embeddings to be searched.
    :param src_ids: The ids of items related to the embeddings.
    :param target_embedding: The embeddings to be searched in.
    :param target_ids: The ids of items related to the target embeddings.
    :param pair_sim_func: A distance function to calculate the distance between embeddings and target embeddings.
        You should ensure, the smaller the distance, the more similar the embeddings are.
    :param device: The device to run the calculation.
    :param batch_size: The batch size of the dataloader.
    :param top_k: The number of top results to be returned.
    :return: A 0-1 matrix. In i-th row, the j-th element denotes whether the id of i-th src embedding is the same as the
     id of target embedding whose similarity to i-th embedding is the j-th largest.
    """
    # Calculate the distance between embeddings and target embeddings.
    target_batch_size = 100000
    src_data = DataLoader(TensorDataset(src_embeddings.to(device), src_ids.to(device)), batch_size=batch_size, shuffle=False)
    target_embedding, target_ids = target_embedding.to(device), target_ids.to(device)
    answer_num = torch.zeros((len(src_ids),), device=device, dtype=torch.int64)
    top_ids = torch.zeros((len(src_ids), top_k), device=device, dtype=torch.int64)
    if verbose:
        src_data = tqdm(src_data)
    for i, (src_embedding, src_id) in enumerate(src_data):
        pair_similarity = torch.zeros((len(src_embedding), len(target_embedding)), device=device, dtype=src_embedding.dtype)
        for j in range(0, len(target_embedding), target_batch_size):
            pair_similarity[:, j:j+target_batch_size] = \
                pair_sim_func(src_embedding, target_embedding[j:j+target_batch_size])
        _, batch_top_idx = torch.topk(pair_similarity, top_k, 1, largest=False)
        top_ids[i*batch_size:(i+1)*batch_size] = batch_top_idx
        answer_num[i*batch_size: (i+1) *batch_size] = (src_id[:, None] == target_ids[None]).sum(dim=1)
    return top_ids, answer_num


@torch.no_grad()
def calculate_mrr(result: torch.Tensor,
                  answer: torch.Tensor,
                  top_k,
                  ignore_first=False) -> float:
    assert top_k + ignore_first <= result.shape[1]
    result = result[:, :top_k + ignore_first]
    if not ignore_first:
        reciprocal_rank = torch.arange(1, top_k + 1, device=result.device).float().reciprocal().reshape([1, -1])
        scores = result * reciprocal_rank
        scores, _ = torch.max(scores, dim=1)
        return torch.mean(scores).cpu().item()
    else:
        reciprocal_rank = torch.arange(0, top_k + 1, device=result.device).float()
        reciprocal_rank[0] = 0.5
        reciprocal_rank = reciprocal_rank.reciprocal()
        scores = result * reciprocal_rank
        scores, _ = torch.topk(scores, 2, dim=1)
        scores = scores.sum(dim=1) - scores.max(dim=1)[0]
        return torch.mean(scores).cpu().item()


@torch.no_grad()
def calculate_topk_hit(result: torch.Tensor,
                       answer_num: torch.Tensor,
                       top_k,
                       ignore_first=False) -> float:
    result = result[:, : top_k + ignore_first]
    if ignore_first:
        return torch.mean((torch.sum(result, dim=1) >= 2).to(torch.float))
    else:
        return torch.mean(torch.max(result, dim=1)[0])

@torch.no_grad()
def calculate_topk_recall(result: torch.Tensor,
                          answer_num: torch.Tensor,
                          top_k,
                          ignore_first=False) -> float:
    assert top_k + ignore_first <= result.shape[1]
    result = result[:, :top_k + ignore_first]
    answer_num = torch.clip(answer_num, 0, top_k + int(ignore_first)) - int(ignore_first)
    TP = result.sum(dim=1) - int(ignore_first)
    TP = torch.clip(TP,0)
    return (TP / answer_num)[answer_num>0].mean().cpu().item()


@torch.no_grad()
def calculate_topk_precision(result: torch.Tensor,
                             answer_num: torch.Tensor,
                             top_k,
                             ignore_first=False) -> float:
    assert top_k + ignore_first <= result.shape[1]
    result = result[:, :top_k + ignore_first]
    TP = result.sum().cpu().item()
    if ignore_first:
        TP -= torch.max(result, dim=1)[0].sum().cpu().item()
    return TP / (result.shape[0] * top_k)


@torch.no_grad()
def calculate_topk_hit(result: torch.Tensor,
                       answer_num: torch.Tensor,
                       top_k,
                       ignore_first=False) -> float:
    assert top_k + ignore_first <= result.shape[1]
    result = result[:, :top_k + ignore_first]
    if ignore_first:
        hit = torch.ge(torch.sum(result, dim=1), 2).float().sum().cpu().item()
    else:
        hit = torch.max(result, dim=1)[0].sum().cpu().item()
    return hit / result.shape[0]


# todo: test the correctness of nDCG
@torch.no_grad()
def calculate_topk_nDCG(result: torch.Tensor,
                        answer_num: torch.Tensor,
                        top_k,
                        ignore_first=False) -> float:
    assert top_k + ignore_first <= result.shape[1]
    result = result[:, :top_k + ignore_first]
    answer_num = torch.clip(answer_num, 0, top_k + int(ignore_first))
    max_answer_num = torch.max(answer_num)
    IDCG_weights = torch.log2(torch.arange(2, max_answer_num + 2, device=result.device)).reciprocal()
    torch.cumsum(IDCG_weights, dim=0, out=IDCG_weights)
    # some function may only has no candidate, we should ignore them when calculating nDCG
    if ignore_first:
        ignored_search_result = (answer_num > 1)
    else:
        ignored_search_result = (answer_num > 0)
    answer_num, result = answer_num[ignored_search_result], result[ignored_search_result]
    # end of filtering process
    IDCG = IDCG_weights[answer_num - 1 - int(ignore_first)]
    if not ignore_first:
        weights = torch.log2(torch.arange(2, top_k + 2, device=result.device)).reciprocal()
        weights = weights.reshape([1, -1])
        DCG = torch.sum(weights * result, dim=1)
        return torch.mean(DCG / IDCG).cpu().item()
    else:
        sorted_scores, ranks = torch.sort(result, dim=1, descending=True, stable=True)
        ignored = ranks[:, 0:1]
        scores = torch.where(torch.bitwise_and(ranks > ignored, sorted_scores == 1), torch.log2(ranks + 1).reciprocal(),
                             0)
        DCG = torch.sum(scores, dim=1)
        return torch.mean(DCG / IDCG).cpu().item()
