from base_layer import FMLayer, MLP
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self,
                 feature_num,
                 embedding_size,
                 hidden_dims=[64,8],
                 dropout=0.0
                 ):
        super().__init__()

        self.fm = FMLayer(feature_num, embedding_size)
        self.mlp = MLP(input_dim=embedding_size * 7,hidden_dims=hidden_dims,dropout=dropout)
        # Final prediction layer
        self.final = nn.Linear(hidden_dims[-1] + 1, 1)  # FM out + MLP out

    def forward(self, x):
        """
        x: [batch_size, num_fields] (indices of categorical features)
        """
        # FM path
        fm_out = self.fm(x)  # [B,1]

        # Deep path
        emb = self.fm.embedding(x)        # [B, F, D]
        deep_in = emb.view(emb.size(0), -1)  # flatten: [B, F*D]
        deep_out = self.mlp(deep_in)         # [B, mlp_units[-1]]

        # Concatenate
        concat = torch.cat([fm_out, deep_out], dim=1)
        logit = torch.sigmoid(self.final(concat))
        return logit
