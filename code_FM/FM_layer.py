"""
inputs:
outputs:
    predicted CTR probability
        → used for CTR prediction / ranking.
    aggregated user embedding vector: composite embedding" or feature-enriched embedding
    aggregated item embedding vector
        → used for recall or nearest-neighbor search.
"""

"""
Variable	Shape	                     Purpose
emb_list	[batch, num_fields, embed]	All embeddings for all features
user_embs	list of [batch,1,embed]	    Only user-related features, for summing/concatenation
item_embs	list of [batch,1,embed]	    Only item-related features, for summing/concatenation
user_emb	[batch, embed]	            Single vector representing the user (sum of embeddings)
item_emb	[batch, embed]	            Single vector representing the item (sum of embeddings)

FM = bias + linear weights + dot-product interactions.
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F

class FMLayer(nn.Module):
    """
    feature_size: total vocab size across all slots
    embedding_size: FM embedding dimension
    user_fields: list of indices in x[] that are user-related (e.g. [0,1,2])
    item_fields: list of indices in x[] that are item-related (e.g. [3,4,5])
    """
    def __init__(self, feature_num, embedding_size,user_fields, item_fields):
        super().__init__()
        self.feature_num = feature_num
        self.embedding_size = embedding_size
        self.user_fields = user_fields
        self.item_fields = item_fields

        # First-order linear weights + bias
        # Index-based lookup
        self.linear = torch.nn.Embedding(feature_num, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        # Embedding for interaction
        self.embedding = torch.nn.Embedding(feature_num, embedding_size)

    def forward(self, x):  # x: [batch_size, num_fields]
        # First-order
        batch_size, num_fields = x.shape

        # embedding lookup get embedding and first-order weight
        linear_part = self.linear(x)  # [batch,feature_num,1]
        emb = self.embedding(x)  # [batch, feature_num,embed]

        # -----------First-Order Linear Term ---------------
        first_order = torch.sum(linear_part, dim=1) + self.bias  # [batch, 1]

        # ----- Second-order interaction -----
        sum_square = torch.sum(emb, dim=1) ** 2  # [batch, embed]
        square_sum = torch.sum(emb ** 2, dim=1)

        second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # [batch,1]

        # Logit Prediction
        logit = torch.sigmoid(first_order + second_order)

        # Recall embeddings
        user_emb = torch.sum(emb[:, self.user_fields,:], dim=1) # [batch, embed_dim]
        item_emb = torch.sum(emb[:, self.item_fields,:],dim=1)

        return logit, user_emb, item_emb
