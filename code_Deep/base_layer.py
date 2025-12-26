import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- FM Layer -------------------
class FMLayer(nn.Module):
    """
    feature_size: total vocab size across all slots
    embedding_size: FM embedding dimension
    user_fields: list of indices in x[] that are user-related
    item_fields: list of indices in x[] that are item-related
    """
    def __init__(self, feature_num, embedding_size):
        super().__init__()
        self.feature_num = feature_num
        self.embedding_size = embedding_size

        # First-order linear weights + bias
        self.linear = torch.nn.Embedding(feature_num, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        # Embedding for interaction
        self.embedding = torch.nn.Embedding(feature_num, embedding_size)

    def forward(self, x):  # x: [batch_size, num_fields]
        # Embedding lookups
        linear_part = self.linear(x)  # [B, F, 1]
        emb = self.embedding(x)  # [B, F, D]

        # First-order term
        first_order = torch.sum(linear_part, dim=1) + self.bias  # [batch, 1]

        # ----- Second-order interaction -----
        sum_square = torch.sum(emb, dim=1) ** 2  # [batch, embed]
        square_sum = torch.sum(emb ** 2, dim=1)

        second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # [batch,1]

        # Logit Prediction
        out = torch.sigmoid(first_order + second_order)

        return out

# ------------------- MLP Layer -------------------
class MLP(nn.Module):
    """
    supports multiple layers, activation, batchnorm, dropout, bias.
    linear layers + activations (and optionally dropout, batchnorm)
    the last layer of an embedding network (like DSSM, FM, DeepFM)
    does neither activation nor batch normalization.
    """
    def __init__(self,input_dim,hidden_dims,dropout=0.0,activation=F.relu,use_bn=True):
        super().__init__()
        self.activation = activation
        self.use_bn = use_bn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # layer dimensions, e.g. [input, h1, h2, ...]
        layer_dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList() if self.use_bn else None

        for i in range(len(layer_dims)-1):
            self.layers.append(nn.Linear(layer_dims[i],layer_dims[i+1]))
            if use_bn and i < len(layer_dims) - 2:# batchnorm expect last
                self.bn.append(nn.BatchNorm1d(layer_dims[i+1]))


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                if self.use_bn:
                    x = self.bn[i](x)
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


# ------------------- DSSM Layer -------------------
class DSSMLayer(nn.Module):
    def __init__(self, feature_num, embedding_size, user_fields, item_fields):
        """
        Deep Structured Semantic Model
        good for recall tasks
        the user tower encodes the user's profile -> user_emb
        the item tower encodes the item's attributes -> item_emb

        """
        super(DSSMLayer, self).__init__()
        self.user_fields = user_fields
        self.item_fields = item_fields

        # Shared embedding layer for all features
        self.embed = nn.Embedding(feature_num, embedding_size)

        # Towers
        self.user_tower = MLP(input_dim=embedding_size*len(self.user_fields),
                              hidden_dims=[64,8],activation=F.relu)

        # Item tower
        self.item_tower = MLP(input_dim=embedding_size * len(item_fields),
                                    hidden_dims=[64,8],activation=F.relu)

    def forward(self, x):
        """
        x: ndarray
        x → embedding lookup → split user/item → flatten
        → MLP(user) / MLP(item)
        → dot product → sigmoid → prediction
        """

        # select user/item columns
        user_features = x[:,self.user_fields]
        item_features = x[:,self.item_fields]

        # embedding lookup and flatten
        user_embs = self.embed(user_features).reshape(x.size(0), -1)  # [batch, num_user_features*embed]
        item_embs = self.embed(item_features).reshape(x.size(0), -1)

        # Feed through towers
        user_emb = self.user_tower(user_embs)
        item_emb = self.item_tower(item_embs)

        # Dot product similarity
        out = torch.sigmoid(torch.sum(user_emb * item_emb, dim=1, keepdim=True))

        return out, user_emb, item_emb


# just test
if __name__ == "__main__":

    B = 4
    in_dim = 16 * 7  #7 fields each with 16-d embeddings
    x = torch.randn(B, in_dim)

    mlp = MLP(input_dim=in_dim, hidden_dims=[64, 8], activation=F.relu, dropout=0.2)
    out = mlp(x)
    print(out.shape)
    print("test_modified")