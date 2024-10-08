import torch
from transformers import BertModel
from binsim.neural.nn.base.model import GraphEmbeddingModelBase
from binsim.neural.nn.siamese.distance import CosineDistance


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


class JTrans(GraphEmbeddingModelBase):
    def __init__(self, pretrained_weights: str, distance_func=None, device=None, dtype=None,
                 freeze_cnt=10):
        super().__init__(CosineDistance())
        self.model = BinBertModel.from_pretrained(pretrained_weights, device_map=device)
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        if freeze_cnt != -1:
            for layer in self.model.encoder.layer[:freeze_cnt]:
                for param in layer.parameters():
                    param.requires_grad = False

    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.JTransSeq

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data import JTransSeqSampleDataset
        return JTransSeqSampleDataset

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in super().parameters())
        token = sum(p.numel() for p in self.model.embeddings.parameters())
        return {'total': total, 'token': token, 'model': total - token}

    def forward(self, samples: torch.Tensor, labels: torch.Tensor, ids: torch.Tensor = None):
        assert labels is None, "JTrans only need samples for triplet loss! No labels are needed!"
        embeddings = self.generate_embedding(*samples)
        embeddings = embeddings.reshape([len(embeddings) // 3, 3, -1])
        anchor, positive, negative = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
        return self.triplet_loss(anchor, positive, negative)

    @staticmethod
    def from_pretrained(pretrained_weights: str, device=None):
        return JTrans(pretrained_weights, device=device, distance_func=CosineDistance())

    def save(self, filename: str):
        self.model.save_pretrained(filename, safe_serialization=True)

    def generate_embedding(self, token_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=token_id, attention_mask=mask).pooler_output


    def triplet_loss(self, anchors: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        good_sim = torch.cosine_similarity(anchors, positive)
        bad_sim = torch.cosine_similarity(anchors, negative)
        loss = (self.margin - (good_sim - bad_sim)).clamp(min=1e-6).mean()
        return loss

    def parameters(self, recurse: bool = True):
        # copied from jTrans
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []

        optimizer_grouped_parameters.extend(
            [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 1e-4,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        )
        return optimizer_grouped_parameters
