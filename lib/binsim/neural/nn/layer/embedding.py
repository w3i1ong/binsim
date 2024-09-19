import torch
from torch.nn import Embedding, Module

class SparseEmbedding(Module):
    def __init__(self, embeddings, index_map, freeze=True, device=None, dtype=None):
        super().__init__()
        if dtype is None:
            dtype = torch.float32
        factory_kwargs = {'device': device, 'dtype': dtype}
        embeddings_tensor = torch.tensor(embeddings, **factory_kwargs)
        embeddings_tensor = torch.cat([embeddings_tensor, torch.zeros(1, embeddings_tensor.size(1), **factory_kwargs)], dim=0)
        self.embeddings = Embedding.from_pretrained(embeddings_tensor, freeze=freeze)
        self.max_index = max(index_map.keys()) + 1
        # the last index is for unknown tokens, so we map all unknown tokens to it
        self.index_map = [len(embeddings_tensor) - 1] * (self.max_index + 1)
        for key, value in index_map.items():
            self.index_map[key] = value
        self.index_map = torch.tensor(self.index_map, device=device, dtype=torch.long)

    @property
    def embedding_dim(self):
        return self.embeddings.embedding_dim


    def forward(self, index):
        new_index = torch.clip(index, 0, self.max_index).to(self.index_map.device)
        new_index = self.index_map[new_index]
        return self.embeddings(new_index)
